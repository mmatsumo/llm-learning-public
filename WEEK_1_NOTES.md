## Week 1 Log
Overview of done activities:
- [x] Transformers paper: re-read chapter 3, and coded the attentio
n scale dot product.
- [x] Watched Kartpathy video and coded along, the first hour
 of the video, until the causal mask math trick of using a triangular matrix and -inf before softmax function
 - [x] Went through concepts of tokens, encoding and decoding

## Artifacts:
- attention.py with scaled dot product
- this log

## Notes:
- Initial input data (like text) can be fragmented into tokens, and those tokens have different IDs, that are coded in the vocabulary vector. Encoding tokens means setting the ID sequence of the input data.
- Decoding is actually the reverse, the output is the sequence IDs that are reversed to text language
- Q, K, V: queries, keys and values. Attention(Q, K, V) is softmax(Q@K.Transpose) @ V. Q shape is (sequence length/ tokens, d_moddel/embedding dimention. K and V have the same shape as Q.

- input text is transformed into tokens (part of token vocabulary)
- Embedding table (IDs idx), embedding is converting discrete tokens into dense vectors, size of embedding dimensions. Embedding matrix are created when training. E.g. matrix(50,000, 768) (vocab, dim embedding)
- Logits will be the scores for next character in the sequence. 
