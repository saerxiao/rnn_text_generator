--This loader prepare the data in the following way:
--original vector: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
--batch size: B = 2
--sequence length: T = 3
--1. truncate the input so that L - 1 is divisible for (B * T)
--2. input is 1 to L-1, target is 2 to L
--3. arrange the input and target to be:
--   1 2 3           2 3 4
--   7 8 9           8 9 10
--   
--   4  5  6         5  6  7
--   10 11 12        11 12 13

require 'torch'
require 'paths'

local Loader = {}
Loader.__index = Loader

local function textToTensor(dir, trainSplit, maxLoad)
  local inputVFile = dir .. '/input_v.t7'
  local vocabFile = dir .. '/vocab.t7'
  if maxLoad > 0 then
    inputVFile = dir .. '/input_v_' .. maxLoad .. '.t7'
    vocabFile = dir .. '/vocab_' .. maxLoad .. '.t7'
  end
  
  local vocab = {}
  local textV = torch.Tensor()
  if not paths.filep(inputVFile) or not paths.filep(vocabFile) then
    local BUFSIZE = 2^13
    local f = assert(io.input(dir .. "/input.txt"))
    local cc = 0
    local alltext = ''
    while true do
      local chunk = f:read(BUFSIZE)
      if not chunk then break end
      cc = cc + chunk:len()
      alltext = alltext .. chunk
      if maxLoad > 0 and cc > maxLoad then break end
    end
    
    textV:resize(cc)
    local i = 1
    alltext:sub(1, math.ceil(cc * trainSplit)):gsub(".", function(c)
      if not vocab[c] then
        table.insert(vocab,c)
        vocab[c] = #vocab
      end
    end)
    i = 1
    alltext:gsub(".", function(c)
      if vocab[c] then
        textV[i] = vocab[c]
      else
        -- for all characters that are not in the vocab, use one number to represent
        textV[i] = #vocab + 1
      end      
      i = i + 1
    end)
  
    torch.save(vocabFile, vocab)
    torch.save(inputVFile, textV)
  else
    vocab = torch.load(vocabFile)
    textV = torch.load(inputVFile) 
  end

  return vocab, textV
end

function Loader.create(opt)
  local self = {}
  setmetatable(self, Loader)
  
  assert(opt.dir)
  assert(opt.seq_len)
  self.batch_size = opt.batch_size or 1
  self.T = opt.seq_len
  local B, T = self.batch_size, self.T
  local train_split = opt.train_split or 0.8
  local validate_split = opt.validate_split or 0.1
--  assert(train_split + validate_split < 1)
  local max_load = opt.max_load or -1
  local vocab, inputV = textToTensor(opt.dir, train_split, max_load)
  self.vocab = vocab
  
  self.dataset = {}
  self.label = {}
  self.nbatch = {}
  local train_end = math.floor(inputV:size(1) * train_split)
  local train_nbatch = math.floor((train_end - 1) / (B * T))
  self.nbatch.train = train_nbatch
  local inputV_train = inputV[{{1, train_end}}]:resize((B * T) * train_nbatch + 1)
  self.dataset.train = inputV_train[{{1, inputV_train:size(1)-1}}]:view(B, -1, T):transpose(1, 2)
  self.label.train = inputV_train[{{2, -1}}]:view(B, -1, T):transpose(1, 2)
  
  local validate_end = math.floor(inputV:size(1) * (train_split + validate_split))
  local validate_nbatch = math.floor((validate_end - (train_end+1) - 1) / (B * T))
  self.nbatch.validate = validate_nbatch
  local inputV_validate = inputV[{{train_end+1, validate_end}}]:resize((B * T) * validate_nbatch + 1)
  self.dataset.validate = inputV_validate[{{1, inputV_validate:size(1)-1}}]:view(B, -1, T):transpose(1,2)
  self.label.validate = inputV_validate[{{2, -1}}]:view(B, -1, T):transpose(1,2)
  
  local test_nbatch = math.floor((inputV:size(1) - (validate_end+1) - 1) / (B * T))
  self.nbatch.test = test.nbatch
  local inputV_test = inputV[{{validate_end+1, -1}}]:resize((B * T) * test_nbatch + 1)
  self.dataset.test = inputV_test[{{1, inputV_test:size(1) - 1}}]:view(B, -1, T):transpose(1,2)
  self.label.test = inputV_test[{{2, -1}}]:view(B, -1, T):transpose(1,2)
  
  return self
end

function Loader:vocabSize()
  return #self.vocab
end

function Loader:iterator(split)
  local B, T, D = self.batch_size, self.T, self:vocabSize()
  local ibatch = 1
  local it = {}
  it.reset = function()
    ibatch = 1
  end
  it.nextBatch = function()
    local data = self.dataset[split][ibatch]
    local data_onehot = torch.Tensor(B, T, D):zero()
    for i = 1, B do
      for j = 1, T do
        data_onehot[i][j][data[i][j]] = 1
      end
    end
    
    local label = self.label[split][ibatch]
    
    ibatch = ibatch + 1
    if ibatch > self.dataset[split]:size(1) then ibatch = 1 end
    
    return data_onehot, label
  end
  return it
end

return Loader