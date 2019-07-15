require 'model.VanillaRnn'
require 'model.LSTM'
require 'model.LSTM1'
require 'nn'

local MultiLayerRnn, parent = torch.class('MultiLayerRnn2', 'nn.Module')

function MultiLayerRnn:__init(wordVecSize, hiddenSize, vocabSize, nLayer, model)
  parent.__init(self)
  local D, H, V = wordVecSize, hiddenSize, vocabSize
  self.net = nn.Sequential()
  self.net:add(nn.LookupTable(V, D))
  self.rnns = {}
  for i = 1, nLayer do
    local rnnInputSize = D
    if (i > 1) then
      rnnInputSize = H
    end
    local rnn
    if model == 'vanilla' then
      rnn = VanillaRnn(rnnInputSize, H)
    elseif model == 'lstm' then
      rnn = LSTM(rnnInputSize, H)
    elseif model == 'lstm1' then
      rnn = LSTM1(rnnInputSize, H)
--      rnn.remember_states = true
    else
      print('Unknown model type')
      error()
    end
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
  end

  self.view1 = nn.View()
  self.net:add(self.view1)
  self.net:add(nn.Linear(H, V))
  self.view2 = nn.View()
  self.net:add(self.view2)
end

function MultiLayerRnn:prepareSampling()
  for _,rnn in pairs(self.rnns) do 
    rnn.remember_states = true
    rnn:resetStates() 
  end
end

-- input B x T
-- output B x T x V
function MultiLayerRnn:updateOutput(input)
  local B, T = input:size(1), input:size(2)
  self.view1:resetSize(T*B, -1)
  self.view2:resetSize(B, T, -1)
  self.output = self.net:forward(input)
  return self.output
end

-- input B x T
-- gradOutput B x T x V
function MultiLayerRnn:backward(input, gradOutput)
  return self.net:backward(input, gradOutput)
end

function MultiLayerRnn:parameters()
  return self.net:parameters()
end