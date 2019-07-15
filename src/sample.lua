require 'torch'
require 'model.MultiLayerRnn1'
require 'model.MultiLayerRnn2'
require 'model.LanguageModel'
require 'cunn'
require 'cutorch'

local function number2char(vocab, number)
  return vocab[number] and vocab[number] or '##REAR##'
end

-- without lookupTable in model
function generate_seq(file, vocabFile, seed, len)
  local vocab = torch.load(vocabFile)
  local s = seed or vocab[#vocab * math.random()]
  local L = len or 200
  local f = torch.load(file)
  local model = f.model
  model:prepareSampling()
  -- TODO: temparay hack
--  model.net.modules[2].remember_states = true
--  model.net.modules[2]:resetStates()
  local w = model.net.modules[1].weight
  local input = w.new(1, 1, #vocab+1):fill(0)
--  local input = torch.zeros(1, 1, #vocab+1)
--  input = input:cuda()
  input[1][1][s] = 1
  local genStr = {}
  table.insert(genStr, number2char(vocab, s))
  for l = 1, L do
    local output = model:forward(input) -- 1 x 1 x V
--    local _,predict = torch.max(output, 3) -- 1 x 1 x 1
    local probs = output:double():exp():squeeze()
    probs:div(torch.sum(probs))
    local predict = torch.multinomial(probs, 1)[1]
    table.insert(genStr,number2char(vocab, predict))
    input:fill(0)
    input[1][1][predict] = 1
  end
  return table.concat(genStr)
end

-- with LookupTable in model
function generate_seq1(file, vocabFile, seed, len)
  local vocab = torch.load(vocabFile)
  local s = seed or vocab[#vocab * math.random()]
  local L = len or 200
  local f = torch.load(file)
  local model = f.model
  model:prepareSampling()
  local w = model.net.modules[1].weight
  local input = w.new(1, 1):fill(0)
  input[1][1] = s
  local genStr = {}
  table.insert(genStr, number2char(vocab, s))
  for l = 1, L do
    local output = model:forward(input) -- 1 x 1 x V
--    local _,predict = torch.max(output, 3) -- 1 x 1 x 1
    local probs = output:double():exp():squeeze()
    probs:div(torch.sum(probs))
    local predict = torch.multinomial(probs, 1)[1]
    table.insert(genStr,number2char(vocab, predict))
    input:fill(0)
    input[1][1] = predict
  end
  return table.concat(genStr)
end