require 'torch'
require 'paths'

-- This loader prepare input data shown as the following example.
-- text: "hello world!", B = 4, T = 3,
-- The first batch, before converting to oneHot, 
-- For clarity, I use the original letter instead of the index in the vocab.
-- input:  hlw
--         eoo  
--         l r 
-- output: eoo
--         l r 
--         lwl
--
-- number of batches per ecpoch: ceil(floor((n-1)/T)/B) -> ceil(floor((12-1)/3)/B) = 1

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

-- attributes
-- dir, required
-- max_load, default the entire file, use -1 to represent
-- batch_size, default 1
-- seq_len, default 10
-- train_split, default 0.8
-- validate_split, default 0.1
function Loader.create(opt)
  local self = {}
  setmetatable(self, Loader)
  
  assert(opt.dir)
  self.batch_size = opt.batch_size or 1
  self.T = opt.seq_len or 10
  local train_split = opt.train_split or 0.8
  local validate_split = opt.validate_split or 0.1
--  assert(train_split + validate_split < 1)
  local max_load = opt.max_load or -1
  local vocab, inputV = textToTensor(opt.dir, train_split, max_load)
  self.vocab = vocab

  local trainEnd = math.ceil(inputV:size(1) * train_split)
  trainEnd = math.floor((trainEnd - 1) / (self.T * self.batch_size)) * (self.T * self.batch_size) + 1
  local validateEnd = math.ceil(inputV:size(1) * (train_split + validate_split))
  self.dataset = {}
  self.dataset.train = inputV[{{1, trainEnd}}]
  if validate_split > 0 then
    local validateLength = validateEnd - trainEnd
    validateEnd = math.floor((validateLength - 1) / (self.T * self.batch_size)) * (self.T * self.batch_size) + 1 + trainEnd
    self.dataset.validate = inputV[{{trainEnd+1, validateEnd}}]
  end
  
  if 1 - train_split - validate_split > 0 then
    local testLength = inputV:size(1) - validateEnd
    local testEnd = math.floor((testLength - 1) / (self.T * self.batch_size)) * (self.T * self.batch_size) + 1 + validateEnd
    self.dataset.test = inputV[{{validateEnd+1, testEnd}}]
  end
  
  self.buffer4mm = torch.Tensor()
  return self
end

local function onehot(vocabSize, y)
  local v = torch.zeros(vocabSize)
    v[y] = 1
  return v
end

function Loader:vocabSize()
  return #self.vocab + 1
end

function Loader:iterator(split)
--  local start = self.batchStart[split]
--  local totalBatch = math.ceil(math.floor(self.dataset[split]:size(1) / self.T) / self.batch_size)
  local totalBatch = self.dataset[split]:size(1) / (self.T * self.batch_size)
--  print('split size: ' .. self.dataset[split]:size(1) .. ' totalBatch: ' .. totalBatch)
  if totalBatch == 0 then
    print('The total number of batches is 0.')
    error()
  end
  local ibatch = 1
  local it = {}
  it.reset = function()
    ibatch = 1
  end
  it.nextBatch = function()
    local inputStart = (self.T * self.batch_size) * (ibatch - 1) + 1
    local inputEnd = (self.T * self.batch_size) * ibatch
    local B = self.batch_size
    if ibatch == totalBatch then
      inputEnd = self.dataset[split]:size(1) - 1
      B = (inputEnd - (self.T * self.batch_size) * (ibatch-1))/ self.T
    end
    
--    print(ibatch .. ' ' .. inputStart .. ' ' .. inputEnd)
    local input = self.dataset[split][{{inputStart, inputEnd}}]:reshape(B, self.T):t()
--    self.buffer4mm:resize(self.T, B, self:vocabSize()) -- all iterator shares the same buffer4mm won't work in multi-threaded?
    local input_onehot = torch.zeros(self.T, B, self:vocabSize())
    for i = 1, self.T do
      for j = 1, B do
        input_onehot[i][j][input[i][j]] = 1
--        self.buffer4mm[i][j] = onehot(self:vocabSize(), input[i][j])
      end
    end
    
    ibatch = ibatch + 1
    if ibatch > totalBatch then
      ibatch = 1
    end
    return input_onehot, self.dataset[split][{{inputStart+1, inputEnd+1}}]:reshape(B, self.T):t()
  end
  return it
end

return Loader