require 'torch'
require 'model.MultiLayerRnn2'
require 'nn'
require 'optim'
require 'pl'

cmd = torch.CmdLine()
cmd:option('-source', 'tinyshakespeare', 'directory for source data')
cmd:option('-train_split', 0.8, 'sequence length')
cmd:option('-validate_split', 0.1, 'sequence length')
cmd:option('-seq_len', 50, 'sequence length')
cmd:option('-batch_size', 10, 'batch size')

cmd:option('-nhidden', 128, 'hidden size')
cmd:option('-nlayers', 2, 'number of rnn layers')
cmd:option('-model', 'lstm1', 'rnn model')

cmd:option('nepochs', 10, 'number of epochs')
cmd:option('-learning_rate',1e-3,'learning rate')

cmd:option('checkpoint_dir', 'checkpoint', 'dir for saving checkpoints')

cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-seed',123,'torch manual random number generator seed')
opt = cmd:parse(arg)

-- load lib for gpu
if opt.gpuid > -1 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then print('package cunn not found!') end
  if not ok2 then print('package cutorch not found!') end
  if ok and ok2 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    cutorch.manualSeed(opt.seed)
  else
    print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
    print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
    print('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

local Loader = require 'Loader4'
local loader = Loader.create{dir="data/" .. opt.source, seq_len=opt.seq_len, batch_size=opt.batch_size, max_load = 1000}
--local loader = Loader.create{dir="data/" .. opt.source, seq_len=opt.seq_len, batch_size=opt.batch_size}

local net = MultiLayerRnn2(loader:vocabSize(), opt.nhidden, loader:vocabSize(), opt.nlayers, opt.model)
local criterion = nn.CrossEntropyCriterion()

-- ship the model to the GPU if desired
if opt.gpuid == 0 then
  net = net:cuda()
  criterion = criterion:cuda()
end

local params, grads = net:getParameters()
--params:uniform(-0.01, 0.01)
params:uniform(0.1, 0.2)

-- ship data to GPU for compute
local function ship2gpu(x, y)
  if opt.gpuid == 0 then
    x = x:cuda()
    y = y:cuda()
  end
  return x,y
end

local trainIter = loader:iterator("train")

-- loader4, return loss, grads
local feval = function(w)
  if w ~= params then
    params:copy(w)
  end
  grads:zero()

  local input, target = trainIter.nextBatch()  -- input: B x T, target: B x T, input/target[b][t]=number in [1,V]
  if opt.gpuid > -1 then
    input, target = ship2gpu(input, target)
  end
  
  local B = input:size(1)
  
  local output = net:forward(input) -- B x T x V
  local loss = 0
  local dloss = output.new() -- B x T x V
  dloss:resize(B, loader.T, loader:vocabSize())
  for t = 1, loader.T do
    loss = loss + criterion:forward(output[{{},t,{}}], target[{{}, t}])
    dloss[{{},t,{}}] = criterion:backward(output[{{},t,{}}], target[{{}, t}])
  end
  
  net:backward(input, dloss)
  
  return loss, grads
end

-- Loader3
local function calAccuracy(split)
  local iter = loader:iterator(split)
  local iter_per_epoch = loader.nbatch[split]
  local hit, ntotal = 0, 0
  for i = 1, iter_per_epoch do
    local input, target = iter.nextBatch()
    if opt.gpuid > -1 then
      input, target = ship2gpu(input, target)
    end
    
    local output = net:forward(input) -- B x T x V
    local _,predict = torch.max(output, 3) 
    if opt.gpuid < 0 then
      predict = predict:squeeze() -- B x T
    end
    for t = 1, target:size(2) do
      for b = 1, target:size(1) do
        if opt.gpuid < 0 then
          if predict:dim() == 1 then
            if predict[b] == target[b][t] then
              hit = hit + 1
            end
          else
            if predict[b][t] == target[b][t] then
              hit = hit + 1
            end
          end          
        else
          if predict[b][t][1] == target[b][t] then
            hit = hit + 1
          end
        end
        
        ntotal = ntotal + 1
      end
    end
  end
  return hit / ntotal
end

-- TODO: something wrong of it
-- gradient check
--local nchecks = 10
--local eps = 1e-3
--for i = 1, 10 do
----  local iparam = math.random(params:size(1))
--  local iparam = i
--  trainIter.reset()
--  local lossOld, gradsOld = feval(params)
--  local analytical = gradsOld[iparam]
--  local oldparam = params[iparam]
--  params[iparam] = oldparam + eps
--  trainIter.reset()
--  local loss1, _ = feval(params)
--  params[iparam] = oldparam - eps
--  trainIter.reset()
--  local loss2, _ = feval(params)
--  local numerical = (loss1 - loss2) / (2 * eps)
--  print('loss1:' .. loss1 .. ' loss2:' .. loss2 .. ' numerical:' .. numerical .. ' analytical:' .. analytical)
--  if numerical ~=0 then
--     print('diff: '.. math.abs((analytical - numerical) / analytical)*100 .. '%')
--  else
--    print('loss1:' .. loss1 .. ' loss2:' .. loss2 .. ' numerical:' .. numerical .. ' analytical:' .. analytical)
--  end
--  params[iparam] = oldparam
--end
--
--trainIter.reset()

if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end
lfs.chdir(opt.checkpoint_dir)
if not path.exists(opt.source) then lfs.mkdir(opt.source) end
lfs.chdir(opt.source)
if not path.exists(opt.model) then lfs.mkdir(opt.model) end
lfs.chdir(opt.model)
local tfolder = 't' .. opt.seq_len
if not path.exists(tfolder) then lfs.mkdir(tfolder) end
lfs.chdir(tfolder)
local hfolder = 'h' .. opt.nhidden
if not path.exists(hfolder) then lfs.mkdir(hfolder) end

local iter_per_epoch = loader.nbatch.train -- for loader4
local optim_opt = {learningRate = opt.learning_rate}
local checkpoint = {}
for i = 1, iter_per_epoch * opt.nepochs do
  local _, loss = optim.adagrad(feval, params, optim_opt)
  
  if i % iter_per_epoch == 0 then
    checkpoint.iter = i
    checkpoint.epoch = i / iter_per_epoch
    if checkpoint.epoch % 3 == 0 then
      optim_opt.learningRate = optim_opt.learningRate / 2
    end
    checkpoint.model = net
    checkpoint.opt = opt
    checkpoint.loss = loss[1]
    checkpoint.train_accuracy = calAccuracy("train")
    checkpoint.validate_accuracy = calAccuracy("validate")
    local savefile = string.format('%s/epoch%.2f_%.4f.t7', hfolder, checkpoint.epoch, loss[1])
    torch.save(savefile, checkpoint)
  end
  
  xlua.progress(i,iter_per_epoch * opt.nepochs)
end
