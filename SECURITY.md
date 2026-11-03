# Security policy

mojo-dllm reads GGUF files, which are untrusted input the moment you download
one. The parser bounds-checks every read against the file size and refuses
implausible counts and tensors that run past the end of the file. The model
loaders refuse tensor types they cannot decode. A crash or out-of-bounds read on a malformed file is a security bug.

## Reporting

Use GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability). Please do not open a public issue.

Include the file or a script that generates it, the command, and what
happened. You will get an acknowledgement within a week.

## Supported versions

Only the latest commit on `main` receives fixes before 1.0.
