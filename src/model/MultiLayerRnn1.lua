require 'model.VanillaRnn'
require 'model.LSTM'
require 'model.LSTM1'
require 'nn'

local MultiLayerRnn, parent = torch.class('MultiLayerRnn1', 'nn.Module')

function MultiLayerRnn:__init(inputSize, hiddenSize, outputSize, nLayer, model)
  parent.__init(self)
  self.net = nn.Sequential()
  self.rnns = {}
  for i = 1, nLayer do
    local rnnInputSize = inputSize
    if (i > 1) then
      rnnInputSize = hiddenSize
    end
    local rnn
    if model == 'vanilla' then
      rnn = VanillaRnn(rnnInputSize, hiddenSize)
    elseif model == 'lstm' then
      rnn = LSTM(rnnInputSize, hiddenSize)
    elseif model == 'lstm1' then
      rnn = LSTM1(rnnInputSize, hiddenSize)
--      rnn.remember_states = true
    else
      print('Unknown model type')
      error()
    end
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
  end
  
  self.V = outputSize
  self.H = hiddenSize
  self.view1 = nn.View()
  self.net:add(self.view1)
  self.net:add(nn.Linear(self.H, self.V))
  self.view2 = nn.View()
  self.net:add(self.view2)
end

function MultiLayerRnn:rnnRememberStates()
  for _,rnn in pairs(self.rnns) do rnn.remember_states = true end
end

-- input T x B x inputSize
-- output T x B x V
function MultiLayerRnn:updateOutput(input)
  local T, B = input:size(1), input:size(2)
  self.view1:resetSize(T*B, -1)
  self.view2:resetSize(T, B, -1)
  self.output = self.net:forward(input)
  return self.output
end

-- input T x B x D
-- gradOutput T x B x V
function MultiLayerRnn:backward(input, gradOutput)
  return self.net:backward(input, gradOutput)
end

local function tinsert(to, from)
  if type(from) == 'table' then
    for i = 1, #from do
      tinsert(to, from[i])
    end
  else
    table.insert(to,from)
  end
end

function MultiLayerRnn:parameters()
  return self.net:parameters()
end