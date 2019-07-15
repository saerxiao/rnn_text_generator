require 'torch'
require 'paths'

-- This loader prepare input data shown as the following example.
-- text: "hello world!", B = 4, T = 3,
-- The first batch, before converting to oneHot, 
-- For clarity, I use the original letter instead of the index in the vocab.
-- input:  hell
--         ello  
--         llo 
-- output: ello
--         llo 
--         lo w
-- The secord batch:
-- input:  o wo 
--          wor
--         worl
-- output:  wor
--         worl
--         orld
-- The third batch:
-- input:  r
--         l
--         d
-- output: l
--         d
--         !
--
-- number of batches per ecpoch: math.ceil((n - t)/B ) -> ceil((12-3)/4) = 3

local Loader = {}
Loader.__index = Loader

local function textToTensor(dir, trainSplit)
  local inputVFile = dir .. '/input_v.t7'
  local vocabFile = dir .. '/vocab.t7'
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
      alltext = alltext .. chunk
      cc = cc + chunk:len()
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
-- batch_size, default 1
-- seq_len, default 10
-- train_split, default 0.8
-- validate_split, default 0.1
function Loader.create(opt)
  local self = {}
  setmetatable(self, Loader)
  
  assert(opt.dir)
  self.batchSize = opt.batch_size or 1
  self.T = opt.seq_len or 10
  local train_split = opt.train_split or 0.8
  local validate_split = opt.validate_split or 0.1
--  assert(train_split + validate_split < 1)
  local vocab, inputV = textToTensor(opt.dir, train_split)
  self.vocab = vocab

  local trainEnd = math.ceil(inputV:size(1) * train_split)
  local validateEnd = math.ceil(inputV:size(1) * (train_split + validate_split))
  self.dataset = {}
  self.dataset.train = inputV[{{1, trainEnd}}]
  if trainEnd+1 < inputV:size(1) then
    self.dataset.validate = inputV[{{trainEnd+1, validateEnd}}]
    if validateEnd + 1 < input:size(1) then
      self.dataset.test = inputV[{{validateEnd+1, inputV:size(1)}}]
    end
  end
  
  self.batchCursor = {}
  self.batchCursor.train = 1
  self.batchCursor.validate = trainEnd + 1
  self.batchCursor.test = validateEnd + 1
  
  self.buffer4m = torch.Tensor()
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
  local cursor = 1
  local it = {}
  it.nextBatch = function()
    local maxB = self.dataset[split]:size(1) + 1 -self.T - cursor
    local B = (self.batchSize < maxB) and self.batchSize or maxB
    local sub = self.dataset[split][{{cursor, cursor+B+self.T-1}}]
--    local m = torch.Tensor(self.T, B+1)
    self.buffer4m:resize(self.T, B+1)
    for i = 1, B+1 do
      self.buffer4m[{{}, i}] = sub[{{i, i+self.T-1}}]
    end
    
    -- update batch cursor
    local nextCursor = cursor + B
    -- at least has one pair of seq of length T
    if nextCursor + 1 + self.T - 1 > self.dataset[split]:size(1) then
      nextCursor = 1
    end 
    cursor = nextCursor
    
--    local mm = torch.Tensor(self.T, B, self:vocab_size())
    self.buffer4mm:resize(self.T, B, self:vocabSize())
    for i = 1, self.T do
      for j = 1, B do
        self.buffer4mm[i][j] = onehot(self:vocabSize(), self.buffer4m[i][j])
      end
    end
  
  return self.buffer4mm, self.buffer4m[{{},{2,B+1}}]
  end
  return it
end

return Loader