import torch
import sys

from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
BertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

BertMaskedLM.eval()


sentence = sys.argv[1]
tokenize_input = BertTokenizer.tokenize(sentence)
tensor_input = torch.tensor([BertTokenizer.convert_tokens_to_ids(tokenize_input)])
predictions = BertMaskedLM(tensor_input)
print(predictions.shape)
loss_fct = torch.nn.CrossEntropyLoss()
loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data
print(6 - loss)
