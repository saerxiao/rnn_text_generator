require 'nn'
require 'torch'

local LSTM, parent = torch.class('LSTM', 'nn.Module')

function LSTM:__init(inputSize, hiddenSize)
  parent.__init(self)
  self.D = inputSize
  self.H = hiddenSize
  
   -- parameters
--  self.weight = torch.Tensor((self.D + self.H) * 4, self.H)
--  self.bias = torch.Tensor(self.H * 4)
  self.U_i = torch.Tensor(self.D, self.H)
  self.W_i = torch.Tensor(self.H, self.H)
  self.bias_i = torch.Tensor(self.H)
  self.U_f = torch.Tensor(self.D, self.H)
  self.W_f = torch.Tensor(self.H, self.H)
  self.bias_f = torch.Tensor(self.H)
  self.U_o = torch.Tensor(self.D, self.H)
  self.W_o = torch.Tensor(self.H, self.H)
  self.bias_o = torch.Tensor(self.H)
  self.U_g = torch.Tensor(self.D, self.H)
  self.W_g = torch.Tensor(self.H, self.H)
  self.bias_g = torch.Tensor(self.H)
  
  -- gradParams
--  self.gradWeight = torch.Tensor((self.D + self.H) * 4, self.H)
--  self.gradBias = torch.Tensor(self.H * 4)
  self.dU_i = torch.Tensor(self.D, self.H)
  self.dW_i = torch.Tensor(self.H, self.H)
  self.dbias_i = torch.Tensor(self.H)
  self.dU_f = torch.Tensor(self.D, self.H)
  self.dW_f = torch.Tensor(self.H, self.H)
  self.dbias_f = torch.Tensor(self.H)
  self.dU_o = torch.Tensor(self.D, self.H)
  self.dW_o = torch.Tensor(self.H, self.H)
  self.dbias_o = torch.Tensor(self.H)
  self.dU_g = torch.Tensor(self.D, self.H)
  self.dW_g = torch.Tensor(self.H, self.H)
  self.dbias_g = torch.Tensor(self.H)
  
  -- hidden states
  self.c = torch.Tensor()
  self.i = torch.Tensor()
  self.f = torch.Tensor()
  self.o = torch.Tensor()
  self.g = torch.Tensor()
end

function LSTM:parameters()
  local w = {}
  local gw = {}
  table.insert(w, self.U_i)
  table.insert(w, self.U_f)
  table.insert(w, self.U_o)
  table.insert(w, self.U_g)
  table.insert(w, self.W_i)
  table.insert(w, self.W_f)
  table.insert(w, self.W_o)
  table.insert(w, self.W_g)
  table.insert(w, self.bias_i)
  table.insert(w, self.bias_f)
  table.insert(w, self.bias_o)
  table.insert(w, self.bias_g)
  
  table.insert(gw, self.dU_i)
  table.insert(gw, self.dU_f)
  table.insert(gw, self.dU_o)
  table.insert(gw, self.dU_g)
  table.insert(gw, self.dW_i)
  table.insert(gw, self.dW_f)
  table.insert(gw, self.dW_o)
  table.insert(gw, self.dW_g)
  table.insert(gw, self.dbias_i)
  table.insert(gw, self.dbias_f)
  table.insert(gw, self.dbias_o)
  table.insert(gw, self.dbias_g)
  
  return w, gw
end

local function initializeBuffer(buffer, input, B, H)
  if buffer == nil then buffer = input.new() end
  if buffer:nElement() ~= B * H then
    buffer:resize(B, H)
  end
  buffer:fill(0)
  return buffer
end

local function initializeReadOnlyBatchVector(buffer, input, B)
  if buffer == nil then buffer = input.new() end
  if buffer:nElement() ~= B then
    buffer:resize(B):fill(1)
  end
  return buffer
end

local function computeGate(gate, t, input, U, s, W, bias, func)
  gate[t]:mm(input, U)
  if s ~= nil then
    gate[t]:addmm(1, s, W)
  end
--  gate[t]:add(bias)
  gate[t]:apply(func)
end

-- input: T x B x D
-- output: T x B x H
function LSTM:updateOutput(input)
  local T = input:size(1)
  local B = input:size(2)
  self.output:resize(T, B, self.H):fill(0)
  self.c:resize(T, B, self.H):fill(0)
  self.i:resize(T, B, self.H):fill(0)
  self.f:resize(T, B, self.H):fill(0)
  self.o:resize(T, B, self.H):fill(0)
  self.g:resize(T, B, self.H):fill(0)
  
  self.buffer_batch_ones = initializeReadOnlyBatchVector(self.buffer_batch_ones, input, B)
  self.bias_i_batch = initializeBuffer(self.bias_i_batch, input, B, self.H)
  self.bias_i_batch:addr(1, self.buffer_batch_ones, self.bias_i)
  self.bias_f_batch = initializeBuffer(self.bias_f_batch, input, B, self.H)
  self.bias_f_batch:addr(1, self.buffer_batch_ones, self.bias_f)
  self.bias_o_batch = initializeBuffer(self.bias_o_batch, input, B, self.H)
  self.bias_o_batch:addr(1, self.buffer_batch_ones, self.bias_o)
  self.bias_g_batch = initializeBuffer(self.bias_g_batch, input, B, self.H)
  self.bias_g_batch:addr(1, self.buffer_batch_ones, self.bias_g)

  for t = 1, T do
    computeGate(self.i, t, input[t], self.U_i, (t > 1) and self.output[t-1] or nil, self.W_i, self.bias_i_batch, torch.sigmoid)
--    print(self.U_i, self.bias_i_batch, self.i)
    computeGate(self.f, t, input[t], self.U_f, (t > 1) and self.output[t-1] or nil, self.W_f, self.bias_f_batch, torch.sigmoid)
--    print(self.U_f, self.bias_f_batch)
    computeGate(self.o, t, input[t], self.U_o, (t > 1) and self.output[t-1] or nil, self.W_o, self.bias_o_batch, torch.sigmoid)
--    print(self.U_o, self.bias_o_batch)
    computeGate(self.g, t, input[t], self.U_g, (t > 1) and self.output[t-1] or nil, self.W_g, self.bias_g_batch, torch.tanh)
--    print(self.U_g, self.bias_g_batch)
    
    self.c[t]:cmul(self.g[t], self.i[t])
    if t > 1 then
      self.c[t]:addcmul(self.c[t-1], self.f[t])
    end
    self.output[t]:tanh(self.c[t]):cmul(self.o[t])
  end
  return self.output
end

-- input T x B x D
-- gradOutput T x B x H
function LSTM:backward(input, gradOutput)
  local T = input:size(1)
  local B = input:size(2)
  self.gradInput:resize(T, B, self.D):fill(0)
  self.dLdsnext = initializeBuffer(self.dLdsnext, input, B, self.H)
  self.dLdcnext = initializeBuffer(self.dLdcnext, input, B, self.H)
  self.buffer1 = initializeBuffer(self.buffer1, input, B, self.H)
  self.buffer_batch_ones = initializeReadOnlyBatchVector(self.buffer_batch_ones, input, B)
  
  for t = T, 1, -1 do
    self.dLds_t = torch.add(gradOutput[t], self.dLdsnext)
    
    local tanhct = torch.tanh(self.c[t])
    self.dLdc_t = torch.cmul(self.dLds_t, self.o[t])
    self.buffer1:fill(1):addcmul(-1, tanhct, tanhct)
    self.dLdc_t:cmul(self.buffer1):add(self.dLdcnext)
    
    self.dLdo_t = torch.cmul(self.dLds_t, tanhct)
    self.buffer1:fill(1):add(-1, self.o[t]):cmul(self.o[t])
    self.dLdo_t:cmul(self.buffer1)
    
    if t > 1 then
      self.dLdf_t = torch.cmul(self.dLdc_t, self.c[t-1])
      self.buffer1:fill(1):add(-1, self.f[t]):cmul(self.f[t])
      self.dLdf_t:cmul(self.buffer1)
      
      self.dU_f:addmm(1, self.dU_f, 1, input[t]:t(), self.dLdf_t)
--      self.dbias_f:addmv(self.dLdf_t:t(), self.buffer_batch_ones)
    end
    
    self.dLdg_t = torch.cmul(self.dLdc_t, self.i[t])
    self.buffer1:fill(1):addcmul(-1, self.g[t], self.g[t])
    self.dLdg_t:cmul(self.buffer1)
    
    self.dLdi_t = torch.cmul(self.dLdc_t, self.g[t])
    self.buffer1:fill(1):add(-1, self.i[t]):cmul(self.i[t])
    self.dLdi_t:cmul(self.buffer1)
    
    self.dU_o:addmm(1, self.dU_o, 1, input[t]:t(), self.dLdo_t)
    self.dU_g:addmm(1, self.dU_g, 1, input[t]:t(), self.dLdg_t)
    self.dU_i:addmm(1, self.dU_i, 1, input[t]:t(), self.dLdi_t)
    
--    self.dbias_o:addmv(self.dLdo_t:t(), self.buffer_batch_ones)
--    self.dbias_g:addmv(self.dLdg_t:t(), self.buffer_batch_ones)
--    self.dbias_i:addmv(self.dLdi_t:t(), self.buffer_batch_ones)
    
    self.gradInput[t]:addmm(self.dLdg_t, self.U_g:t()):addmm(self.dLdo_t, self.U_o:t()):addmm(self.dLdi_t, self.U_i:t())
    if t > 1 then
      self.gradInput[t]:addmm(self.dLdf_t,self.U_f:t())
    end
   
    if t > 1 then
      self.dW_g:addmm(self.output[t-1]:t(), self.dLdg_t)
      self.dW_f:addmm(self.output[t-1]:t(), self.dLdf_t)
      self.dW_o:addmm(self.output[t-1]:t(), self.dLdo_t)
      self.dW_i:addmm(self.output[t-1]:t(), self.dLdi_t)
    end
    
    self.dLdsnext:mm(self.dLdg_t, self.W_g:t()):addmm(self.dLdi_t, self.W_i:t()):addmm(self.dLdo_t, self.W_o:t())
    if t > 1 then
      self.dLdsnext:addmm(self.dLdf_t, self.W_f:t())
    end
    self.dLdcnext:cmul(self.dLdc_t, self.f[t])
  end
  return self.gradInput
end

