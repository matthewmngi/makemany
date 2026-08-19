# makemore

makes more of the things you give it
- treats each example as a sequence of characters
- character level model because it predicts the next character in a sequence given a previous sequence of characters
- components: MLP, RNN, GRU, and eventually, transformer

- each word gives some insights as to the patterns around each character

## bigram lm
- works w/ two characters at a time
- given one character, predict the one after it
- very simple and weak