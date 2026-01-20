SESSION=train
if ! tmux has-session -t $SESSION 2>/dev/null; then
  tmux new -s $SESSION
else
  tmux attach -t $SESSION
fi

# Detach from the session by pressing Ctrl + B, then D