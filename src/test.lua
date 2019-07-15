require 'inspectCheckpoint'
require 'nn'
require 'sample'
require 'lfs'

criterion = nn.CrossEntropyCriterion()
B, T ,V = 2, 2, 2
output = torch.ones(B,T,V)*0.5
output[{{},2,1}]:fill(1)
target = torch.ones(B,T)

print(output[{{},1,{}}])
print(output[{{},2,{}}])
l = criterion:forward(output:view(B*T,-1), target:view(B*T,1))
dl = criterion:backward(output:view(B*T,-1), target:view(B*T,1)):clone()

l1 = 0
dl1 = torch.Tensor(B,T,V)
for t = 1, T do
  l1 = l1 + criterion:forward(output[{{},t,{}}], target[{{},t}]) / T
  dl1[{{},t,{}}] = (criterion:backward(output[{{},t,{}}], target[{{},t}]) / T):clone()
end

--dl1 = torch.Tensor(B,T,V)
--for t = 1, T do
--  dl1[{{},t,{}}] = criterion:backward(output[{{},t,{}}], target[{{},t}])
--  print(dl1)
--end

print(l)
print(dl)
print(l1)
print(dl1)

--local inputVFile = "data/chtest/input_v.t7"
--local vocabFile = "data/chtest/vocab.t7"
--local inputv = torch.load(inputVFile)
--local vocab = torch.load(vocabFile)
--print(inputv)
--print(vocab)


--local x = 1
--local w = 0.1
--local eps = 0.001
--local u = function(x,w)
--  return torch.sigmoid(x*w)
--end
--local u0 = u(x,w)
--local y0 = torch.tanh(u0)
--local u1 = u(x,w+eps)
--local y1 = torch.tanh(u1)
--local u2 = u(x,w-eps)
--local y2 = torch.tanh(u2)
--local numerical = (y1 - y2) / (2*eps)
----local analytical = (1 - y0*y0)*x
--local analytical = (1 - y0*y0)*u0*(1-u0)*x
--print(u1)
--print(u2)
--print(y1)
--print(y2)
--print(numerical)
--print(analytical)
--print(math.abs((numerical - analytical) / numerical)*100 .. '%')


--local a = torch.ones(2,3,4)
----local v1 = nn.View(1,1,-1):setNumInputDims(3)
----local v1 = nn.View(-1):setNumInputDims(1)
--local v1 = nn.View()
--v1:resetSize(6,-1)
--local a1 = v1:forward(a)
--print(a1:size())
--local v2 = nn.View()
--v2:resetSize(2,3,-1)
--local a2 = v2:forward(a1)
--print(a2:size())

--function compute(a, b, func)
--  a = func(a)
--  return a
--end
--
--t = 1
--a = torch.ones(1)
--print(compute(a, (t>1) and 1 or nil, torch.tanh))
--plot_accuracy("checkpoint/tinyshakespeare/vanilla/t100/h128")
--plot_accuracy("checkpoint/tinyshakespeare/lstm/t100")
--plot_accuracy("checkpoint/tinyshakespeare/lstm2/t50/h128")

--local sample = generate_seq("checkpoint/tinyshakespeare/lstm1/t100/h256/epoch10.00_332.5735.t7", "data/tinyshakespeare/vocab.t7",1,1000)
--local sample = generate_seq1("checkpoint/tinyshakespeare/lstm2/t50/h128/epoch10.00_160.3124.t7", "data/tinyshakespeare/vocab_1000.t7",1,1000)
--print(sample)

--a = 1
--assert(a)
--
--local Loader = require 'Loader2'
----local loader = Loader.create{dir="data/" .. "test", seq_len=3, batch_size=4, train_split=1, validate_split=0}
--local loader = Loader.create{dir="data/" .. "tinyshakespeare", seq_len=10, batch_size=20, max_load=1000}
--local iter = loader:iterator("validate")
--while true do
--  local input, target = iter.nextBatch()
----  print(input)
--  print(target)
--end

--
--local BUFSIZE = 2^13
--local file = "data/tinyshakespeare/input.txt"
--local f = assert(io.input(file))
--local cc = 0
--while true do
--  local chunk = f:read(BUFSIZE)
--  if not chunk then break end
--  cc = cc + string.len(chunk)
--end
--print(cc)
--
--local text = ''
--text = text .. 'hello world!\n'
--print(text)
--local vocab = {}
--local cc = 0
--text:gsub(".", function(c)
--    cc = cc + 1
--    if not vocab[c] then
--      table.insert(vocab,c)
--      vocab[c] = #vocab
--    end
--end)
--
--local V = #vocab
--local text_v = torch.Tensor(cc)
--local i = 1
--text:gsub(".", function(c)
--  text_v[i] = vocab[c]
--  i = i + 1
--end)
--
--print(text_v)
--local T, B = 3, 4
--local iB = 1
--local maxB = cc + 1 - T - iB
--local nB = (B < maxB) and B or maxB
--local sub = text_v[{{iB, iB+nB+T-1}}]
--local m = torch.Tensor(T, nB+1)
--for i = 1, nB+1 do
--  m[{{}, i}] = sub[{{i, i+T-1}}]
--end
--print(m)
--torch.save("data/tinyshakespeare/vocab.t7", vocab)
--local svocab = torch.load("data/tinyshakespeare/vocab.t7")
--print(svocab)
--
--local function onehot(y)
--  local v = torch.zeros(V)
--  v[y] = 1
--  return v
--end
--
--local mm = torch.Tensor(T, nB+1, V)
--for i = 1, T do
--  for j = 1, nB+1 do
--    mm[i][j] = onehot(m[i][j])
--  end
--end
--
--print(mm[1])
