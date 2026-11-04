# Records the raw take with charmbracelet/vhs:  vhs assets/demo.tape
# assets/README.md has the ffmpeg step that retimes it into assets/demo.gif.
# Needs build/mojo-dllm and the LLaDA GGUF at ~/models/llada/.

Output "/tmp/demo-raw.gif"

Set Shell "bash"
Set FontSize 18
Set Width 1100
Set Height 420
Set Padding 24
Set Theme "Builtin Dark"
Set TypingSpeed 35ms

Hide
Type "cd ~/Desktop/mojo-dllm && ln -sf ~/models/llada/LLaDA-8B-Instruct.Q4_K_M.gguf /tmp/llada-8b.gguf && clear"
Enter
Sleep 500ms
Show

Type `build/mojo-dllm run --model /tmp/llada-8b.gguf --prompt "Write a haiku about compilers." --max-tokens 32 --steps 16 --visual`
Sleep 3s
Enter
Sleep 34s
Sleep 3s
