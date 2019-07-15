require 'model.VanillaRnn'
require 'model.LSTM'

local MultiLayerRnn, parent = torch.class('MultiLayerRnn', 'nn.Module')

function MultiLayerRnn:__init(inputSize, hiddenSize, outputSize, nLayer, model)
  parent.__init(self)
  self.rnn = nn.Sequential()
  for i = 1, nLayer do
    local rnnInputSize = inputSize
    if (i > 1) then
      rnnInputSize = hiddenSize
    end
    if model == 'vanilla' then
      self.rnn:add(VanillaRnn(rnnInputSize, hiddenSize))
    elseif model == 'lstm' then
      self.rnn:add(LSTM(rnnInputSize, hiddenSize))
    else
      print('Unknown model type')
      error()
    end
  end
  self.V = outputSize
  self.H = hiddenSize
  self.weight = torch.Tensor(hiddenSize, self.V) -- W_hy
  self.bias = torch.Tensor(self.V)
  self.gradWeight = torch.Tensor(hiddenSize, self.V) -- B_hy
  self.gradBias = torch.Tensor(self.V)
  
  self.rnnOutput = torch.Tensor()

--  self.buffer1 = torch.Tensor()
--  self.buffer4bias = torch.Tensor()
end

-- input T x B x inputSize
-- output T x B x V
function MultiLayerRnn:updateOutput(input)
  local T, B = input:size(1), input:size(2)
  self.rnnOutput = self.rnn:forward(input) -- T x B x H
  self.output:resize(T, B, self.V):fill(0)
--  local buffer1 = torch.ones(B)
--  local bias = torch.Tensor(B, self.V):fill(0)
--  bias:addr(1, buffer1, self.bias)
  self.buffer1 = self.buffer1 or input.new()
  if self.buffer1:nElement() ~= B then 
    self.buffer1:resize(B):fill(1)
  end
  self.buffer4bias = self.buffer4bias or input.new()
  if self.buffer4bias:nElement() ~= B*self.V then 
    self.buffer4bias:resize(B, self.V)
  end
  self.buffer4bias:fill(0)
  self.buffer4bias:addr(1,self.buffer1, self.bias)
  for t = 1, T do
--    self.output[t]:addmm(0, self.output[t], 1, self.rnnOutput[t], self.weight) -- B x V
    self.output[t]:addmm(1, self.rnnOutput[t], self.weight) -- B x V
    self.output[t]:add(self.buffer4bias)
  end
  return self.output
end

-- input T x B x D
-- gradOutput T x B x V
function MultiLayerRnn:backward(input, gradOutput)
  local T = input:size(1)
  local B = input:size(2)
  self.dLdh = self.dLdh or input.new()
  if self.dLdh:nElement() ~= T*B*self.H then self.dLdh:resize(T, B, self.H) end
  self.dLdh:fill(0)
--  local dLdh = torch.Tensor(T, B, self.H):fill(0)
--  local buffer1 = torch.ones(B)
  self.buffer1 = self.buffer1 or input.new()
  if self.buffer1:nElement() ~= B then 
    self.buffer1:resize(B):fill(1)
  end
  for t = T, 1, -1 do
    self.gradWeight:addmm(1, self.gradWeight, 1, self.rnnOutput[t]:t(), gradOutput[t]) -- H x V
    self.gradBias:addmv(1, self.gradBias, 1, gradOutput[t]:t(), self.buffer1) -- V
--    dLdh[t]:addmm(0, dLdh[t], 1, gradOutput[t], self.weight:t()) -- B x H
    self.dLdh[t]:addmm(1, gradOutput[t], self.weight:t()) -- B x H
  end
  return self.rnn:backward(input, self.dLdh)
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
  local w = {}
  local gw = {}
  local rnnparam, rnngparam = self.rnn:parameters()
  tinsert(w, rnnparam)
  tinsert(gw, rnngparam)
--  local param, gparam = parent:parameters()
  tinsert(w, self.weight)
  tinsert(w, self.bias)
  tinsert(gw, self.gradWeight)
  tinsert(gw, self.gradBias)
  return w, gw
end


