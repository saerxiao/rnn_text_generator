require 'nn'

local VanillaRnn, parent = torch.class('VanillaRnn1', 'nn.Module')

function VanillaRnn:__init(inputSize, hiddenSize)
  parent.__init(self)
  self.D = inputSize
  self.H = hiddenSize
  
  -- parameters
  self.weight = torch.Tensor(self.D + self.H, self.H)
  self.bias = torch.Tensor(self.H)
--  self.W_xh = torch.Tensor(self.D, self.H)
--  self.W_hh = torch.Tensor(self.H, self.H)
--  self.B_xh = torch.Tensor(self.H)
  
  -- gradParams
  self.gradWeight = torch.Tensor(self.D + self.H, self.H)
  self.gradBias = torch.Tensor(self.H)
--  self.dW_xh = torch.Tensor(self.D, self.H)
--  self.dW_hh = torch.Tensor(self.H, self.H)
--  self.dB_xh = torch.Tensor(self.H)
  
  -- hidden state
  self.h = torch.Tensor()
  
  -- buffer for computing
--  self.buffer1 = torch.Tensor()
--  self.buffer4bias = torch.Tensor()
end

-- input T x B x D
-- output T x B x H
function VanillaRnn:updateOutput(input)
  local B = input:size(2)
  local T = input:size(1)
  self.h:resize(T, B, self.H):fill(0)
  self.hbuffer = self.hbuffer or input.new()
  if self.hbuffer:nElement() == 0 then self.hbuffer:resize(B, self.H):fill(0) end
  local prevH = self.hbuffer
  if self.output:nElement() > 0 then
    prevH = self.output[{{T, T}}][1]
  else
    self.output:resize(T, B, self.H)
  end
--  local prevH = self.output:nElement() == 0 and self.output[{{T}}] or 
--  local buffer1 = torch.ones(B)
--  local bias = torch.Tensor(B, self.H):fill(0)
--  bias:addr(1, buffer1, self.bias)
  self.buffer1 = self.buffer1 or input.new()
  if self.buffer1:nElement() ~= B then 
    self.buffer1:resize(B):fill(1)
  end
  self.buffer4bias = self.buffer4bias or input.new()
  if self.buffer4bias:nElement() ~= B*self.H then 
    self.buffer4bias:resize(B, self.H)
  end
  self.buffer4bias:fill(0):addr(1, self.buffer1, self.bias)
  for t = 1, T do
    self.h[t]:addmm(input[t], self.weight[{{1, self.D}}])
    self.h[t]:addmm(prevH, self.weight[{{self.D+1, self.D+self.H}}])
    self.h[t]:add(self.buffer4bias)
    self.output[t] = torch.tanh(self.h[t])
    prevH = self.output[t]
  end
  return self.output
end

-- gradOutput T x B x H
function VanillaRnn:backward(input, gradOutput)
  local T = input:size(1)
  local B = input:size(2)
  self.gradInput:resize(T, B, self.D):fill(0)
  self.dLdhnext = self.dLdhnext or input.new()
  if self.dLdhnext:nElement() ~= B*self.H then self.dLdhnext:resize(B, self.H) end
  self.dLdhnext:fill(0)
  
  self.buffer1 = self.buffer1 or input.new()
  if self.buffer1:nElement() ~= B*self.H then self.buffer1:resize(B, self.H) end
  
  self.buffer2 = self.buffer2 or input.new()
  if self.buffer2:nElement() ~= B then self.buffer2:resize(B):fill(1) end
  
  for t = T, 1, -1 do
    self.dLdh_t = torch.add(gradOutput[t], self.dLdhnext) -- B x H
    self.buffer1:fill(1):addcmul(-1, self.h[t], self.h[t])
    self.dLdh_t:cmul(self.buffer1) -- element-wise multiplication B x H 
    self.gradWeight[{{1, self.D}}]:addmm(1, self.gradWeight[{{1, self.D}}], 1, input[t]:t(), self.dLdh_t)   -- D x H 
    if t > 1 then
      self.gradWeight[{{self.D+1, self.D+self.H}}]:addmm(1, self.gradWeight[{{self.D+1, self.D+self.H}}], 1, self.output[t-1]:t(), self.dLdh_t) -- H x H
    end
    self.gradBias:addmv(1, self.gradBias, 1, self.dLdh_t:t(), self.buffer2)
    self.gradInput[t]:addmm(1, self.dLdh_t, self.weight[{{1, self.D}}]:t())  -- B x D
    self.dLdhnext:addmm(1, self.dLdh_t, self.weight[{{self.D+1, self.D+self.H}}]:t()) -- B x H
  end
  return self.gradInput
end