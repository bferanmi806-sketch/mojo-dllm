"""A small flag parser: `--name value` pairs plus boolean `--flag`s."""

from std.collections import Dict


struct Args(Movable):
    var command: String
    var values: Dict[String, String]
    var flags: List[String]
    var positional: List[String]
    var unknown: List[String]
    var missing_value: String

    def __init__(
        out self,
        argv: List[String],
        value_flags: List[String],
        boolean_flags: List[String],
    ):
        """Parses `argv[2:]` against the flags the command declares.

        A flag the command does not declare goes to `unknown` and takes no
        value. `--help` and `-h` set the boolean flag `help` for every command.
        A value flag in last position stops the parse and is named in
        `missing_value`; the caller decides whether help outranks that.
        """
        self.values = Dict[String, String]()
        self.flags = List[String]()
        self.positional = List[String]()
        self.unknown = List[String]()
        self.missing_value = String("")
        self.command = argv[1] if len(argv) > 1 else String("")
        var i = 2
        while i < len(argv):
            var a = argv[i]
            if a == "--help" or a == "-h":
                self.flags.append("help")
                i += 1
            elif a.startswith("--"):
                var name = String(a[byte=2:])
                var is_bool = False
                for f in boolean_flags:
                    if f == name:
                        is_bool = True
                if is_bool:
                    self.flags.append(name)
                    i += 1
                    continue
                var takes_value = False
                for f in value_flags:
                    if f == name:
                        takes_value = True
                if not takes_value:
                    self.unknown.append(name)
                    i += 1
                    continue
                if i + 1 >= len(argv):
                    self.missing_value = name
                    break
                self.values[name] = argv[i + 1]
                i += 2
            else:
                self.positional.append(a)
                i += 1

    def has(self, name: String) -> Bool:
        if name in self.values:
            return True
        for f in self.flags:
            if f == name:
                return True
        return False

    def get(self, name: String) raises -> String:
        if name not in self.values:
            raise Error("missing required flag --" + name)
        return self.values[name]

    def get_or(self, name: String, default: String) raises -> String:
        if name in self.values:
            return self.values[name]
        return default

    def int_or(self, name: String, default: Int) raises -> Int:
        if name in self.values:
            return Int(self.values[name])
        return default

    def float_or(self, name: String, default: Float64) raises -> Float64:
        if name in self.values:
            return Float64(self.values[name])
        return default


def edit_distance(a: String, b: String) -> Int:
    """The Levenshtein distance between two strings, counted in bytes."""
    var x = a.as_bytes()
    var y = b.as_bytes()
    var prev = List[Int]()
    for j in range(len(y) + 1):
        prev.append(j)
    for i in range(len(x)):
        var cur = List[Int]()
        cur.append(i + 1)
        for j in range(len(y)):
            var cost = 0 if x[i] == y[j] else 1
            cur.append(min(prev[j] + cost, min(prev[j + 1], cur[j]) + 1))
        prev = cur^
    return prev[len(y)]


def suggest_flag(name: String, declared: List[String]) -> String:
    """The declared flag closest to `name` (distance at most 2), or "" if none.

    The first declared flag wins a tie.
    """
    var best = String("")
    var best_distance = 3
    for flag in declared:
        var d = edit_distance(name, flag)
        if d < best_distance:
            best_distance = d
            best = flag
    return best


def unknown_flag_error(
    command: String, name: String, declared: List[String]
) -> String:
    var text = "unknown flag --" + name + " for '" + command + "'"
    var guess = suggest_flag(name, declared)
    if guess.byte_length() > 0:
        text += " (did you mean --" + guess + "?)"
    return text + "\nsee: mojo-dllm " + command + " --help"


def env_name(flag: String) -> String:
    """The environment variable a flag reads its value from.

    `MOJO_DLLM_` plus the flag name upper-cased with `-` turned into `_`, so
    `max-tokens` gives `MOJO_DLLM_MAX_TOKENS`. Flag names are ASCII (they are
    literals in `command_spec`), so the case map walks bytes rather than
    relying on a Unicode-aware upper.
    """
    var out = "MOJO_DLLM_"
    for b in flag.as_bytes():
        var c = Int(b)
        if c >= 97 and c <= 122:  # a-z -> A-Z
            c -= 32
        elif c == 45:  # '-' -> '_'
            c = 95
        out += chr(c)
    return out^


def env_flags() -> List[String]:
    """The ten value flags of `run` the environment may fill, in order.

    Booleans (`--json`, `--visual`, …), `--prompt` and `--visual-interval`
    get no variable: the first take no value to read and the other two are
    the request itself, not a run setting. Each flag here is a declared
    value flag of `run`, which `test_env_flags_are_value_flags_of_run`
    cross-checks against `command_spec`.
    """
    return List[String](
        [
            "model",
            "max-tokens",
            "steps",
            "block-length",
            "seed",
            "temperature",
            "threads",
            "device",
            "remasking",
            "threshold",
        ]
    )


def with_env(
    raw: List[String],
    value_flags: List[String],
    boolean_flags: List[String],
    env: Dict[String, String],
) raises -> Tuple[List[String], List[String]]:
    """Fill absent `env_flags()` from `env`, returning the new argv and names.

    Builds `Args(raw, …)` and, for each flag in `env_flags()` that the parse
    reports absent and whose `env[env_name(flag)]` is non-empty, collects the
    pair `--flag value`. The pairs go right after the command word
    (`raw[0]`, `raw[1]`, the pairs, then `raw[2:]`), so a value flag that was
    last on the command line still trips its `needs a value` error.

    "Absent" comes from `Args`, not a text scan of `raw`: in
    `run --prompt --steps`, `--steps` is the *value* of `--prompt`, and a scan
    would mistake it for the flag being present. An empty value counts as
    unset, as `NO_COLOR` does. The second element names the flags that came
    from the environment, so a caller can report them.
    """
    var a = Args(raw, value_flags, boolean_flags)
    var argv = List[String]()
    var names = List[String]()
    argv.append(raw[0])
    if len(raw) > 1:
        argv.append(raw[1])
    for flag in env_flags():
        if a.has(flag):
            continue
        var value = env.get(env_name(flag), "")
        if value.byte_length() == 0:
            continue
        argv.append("--" + flag)
        argv.append(value)
        names.append(flag)
    for i in range(2, len(raw)):
        argv.append(raw[i])
    return (argv^, names^)


def parse_csv_ints(s: String) raises -> List[Int]:
    var out = List[Int]()
    for part in s.split(","):
        var p = String(part).strip()
        if p.byte_length() > 0:
            out.append(Int(p))
    return out^
