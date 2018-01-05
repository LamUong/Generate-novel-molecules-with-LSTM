import torch
import torch.nn as nn
from torch.autograd import Variable
from data_loading import *
'''
the model
'''
class generative_model(nn.Module):
    def __init__(self, vocabs_size, hidden_size, output_size, embedding_dimension, n_layers):
        super(generative_model, self).__init__()
        self.vocabs_size = vocabs_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dimension = embedding_dimension
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocabs_size, embedding_dimension)
        self.rnn = nn.LSTM(embedding_dimension, hidden_size, n_layers, dropout = 0.2)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        batch_size = input.size(0)
        input = self.embedding(input)
        output, hidden = self.rnn(input.view(1, batch_size, -1), hidden)
        output = self.linear(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        hidden=(Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return hidden

data,vocabs=load_data()
vocabs = list(vocabs)
vocabs_size = len(vocabs)
output_size = len(vocabs)
batch_size = 128
cuda = True
hidden_size = 1024
embedding_dimension =  248
n_layers=3
lr = 0.005
n_batches = 200000
print_every = 100
plot_every = 10
save_every = 1000
end_token = ' '

print("processing batches ...")
train_batches,val_batches = process_data_to_batches(data,batch_size,vocabs,cuda)
print("finish processing batches")


model = generative_model(vocabs_size,hidden_size,output_size,embedding_dimension,n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

def propagation(inp, target, mode):
    batch_size = inp.size(0)
    sequence_length = inp.size(1)
    hidden = model.init_hidden(batch_size)
    if cuda:
        hidden = (hidden[0].cuda(), hidden[1].cuda())
    if mode=='train':
        model.zero_grad()
    loss = 0
    for c in range(sequence_length):
        output, hidden = model(inp[:,c], hidden)
        loss += criterion(output, target[:,c])
    if mode=='train':
        loss.backward()
        optimizer.step()
    return loss.data[0]/sequence_length


def evaluate(prime_str='!', temperature=0.4):
    max_length = 200
    inp = Variable(tensor_from_chars_list(prime_str,vocabs,cuda)).cuda()
    batch_size = inp.size(0)
    hidden = model.init_hidden(batch_size)
    if cuda:
        hidden = (hidden[0].cuda(), hidden[1].cuda())
    predicted = prime_str
    while True:
        output, hidden = model(inp, hidden)
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        # Add predicted character to string and use as next input
        predicted_char = vocabs[top_i]

        if predicted_char ==end_token or len(predicted)>max_length:
            return predicted

        predicted += predicted_char
        inp = Variable(tensor_from_chars_list(predicted_char,vocabs,cuda)).cuda()
    return predicted

start = time.time()
all_losses = []
loss_avg = 0

for batch in range(1,n_batches+1):
    inp, target = get_random_batch(train_batches)
    loss = propagation(inp, target, 'train')      
    loss_avg += loss
    if batch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), batch, batch / n_batches * 100, loss))
        print(evaluate('!'), '\n')
    if batch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0
    if batch %save_every ==0:
        torch.save(model.state_dict(), 'mytraining.pt')




