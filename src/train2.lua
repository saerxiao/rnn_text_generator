require 'torch'
require 'model.LanguageModel'
require 'nn'
require 'optim'
require 'pl'

cmd = torch.CmdLine()
cmd:option('-source', 'tinyshakespeare', 'directory for source data') --tinyshakespeare, qts, qsc
cmd:option('-train_split', 0.8, 'sequence length')
cmd:option('-validate_split', 0.1, 'sequence length')
cmd:option('-seq_len', 64, 'sequence length') -- 50
cmd:option('-batch_size', 100, 'batch size')

cmd:option('-model', 'lstm2', 'rnn model')
-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'lstm')
cmd:option('-wordvec_size', 64) -- 16, 64
cmd:option('-rnn_size', 128) -- 128, 256
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0)
cmd:option('-batchnorm', 0)
cmd:option('onehot', false)

cmd:option('nepochs', 60, 'number of epochs')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-lr_decay_every', 5)
cmd:option('-lr_decay_factor', 0.5)
cmd:option('-grad_clip', 5)

cmd:option('-checkpoint_every', 10)
cmd:option('-save_every', 1000)
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

local Loader, loader = nil, nil
if opt.onehot then
  Loader = require 'Loader3'
  --local loader = Loader.create{dir="data/" .. opt.source, seq_len=opt.seq_len, batch_size=opt.batch_size, max_load = 1000}
  loader = Loader.create{dir="data/" .. opt.source, seq_len=opt.seq_len, batch_size=opt.batch_size}
else
  Loader = require 'Loader4'
  --local loader = Loader.create{dir="data/" .. opt.source, seq_len=opt.seq_len, batch_size=opt.batch_size, max_load = 1000}
  loader = Loader.create{dir="data/" .. opt.source, seq_len=opt.seq_len, batch_size=opt.batch_size}
end


opt.idx_to_token = loader.vocab
local net = nn.LanguageModel(opt)
local criterion = nn.CrossEntropyCriterion()

local B, T = opt.batch_size, opt.seq_len

-- ship the model to the GPU if desired
if opt.gpuid == 0 then
  net = net:cuda()
  criterion = criterion:cuda()
end

local params, grads = net:getParameters()
--params:uniform(-0.01, 0.01)
--params:uniform(0.1, 0.2)

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

  local input, target = trainIter.nextBatch()  -- input: onehot - B x T x D, not onehot - B x T, target: B x T, target[b][t]=number in [1,V]
  if opt.gpuid > -1 then
    input, target = ship2gpu(input, target)
  end
  
  local B = input:size(1)
  local T = loader.T
  local V = loader:vocabSize()
  local output = net:forward(input) -- B x T x V
  
  -- This is correct, till Line 121
--  local loss1 = 0
--  local dloss = output.new() -- B x T x V
--  dloss:resize(B, T, V)
--  for t = 1, T do
--    loss1 = loss1 + criterion:forward(output[{{},t,{}}], target[{{}, t}]) / T
--    dloss[{{},t,{}}] = (criterion:backward(output[{{},t,{}}], target[{{}, t}]) / T):clone()
--  end

-- This is wrong
-- two mistakes: 1. should compute backward for each t right after the forward, not after computing the forward for all the T,
-- because in that case, the state of the criterion will be for the last T, so we actually use the state for the last T to compute the backward
-- for each t
-- 2. we need to clone the result of the back, other wise all the t would be the result for the last t
--  for t = 1, T do
--    dloss[{{},t,{}}] = criterion:backward(output[{{},t,{}}], target[{{}, t}])
--  end
--  net:backward(input, dloss)
  
  -- Use the Criterion to compute loss; we need to reshape the scores to be
  -- two-dimensional before doing so. Annoying.
  local scores_view = output:view(B * T, -1)
  local y_view = target:view(B * T)
  local loss = criterion:forward(scores_view, y_view)

  -- Run the Criterion and model backward to compute gradients, maybe timing it
  local grad_scores = criterion:backward(scores_view, y_view):view(B, T, -1)
  net:backward(input, grad_scores)
  
  if opt.grad_clip > 0 then
    grads:clamp(-opt.grad_clip, opt.grad_clip)
  end
  
  return loss, grads
end

-- Loader3/4
local function calAccuracy(split)
  local iter = loader:iterator(split)
  local iter_per_epoch = loader.nbatch[split]
  local hit, ntotal = 0, 0
  local loss = 0
  for i = 1, iter_per_epoch do
    local input, target = iter.nextBatch()
    if opt.gpuid > -1 then
      input, target = ship2gpu(input, target)
    end
    
    target = target:view(B * T)
    
    local output = net:forward(input):view(B * T, -1) -- B x T x V
    local _, predict = output:max(2)
    predict = predict:squeeze():type(target:type())
    hit = hit + torch.eq(target, predict):sum()
    
    loss = loss + criterion:forward(output, target)
    
--    if opt.gpuid < 0 then
--      predict = predict:squeeze() -- B x T
--    end
--    for t = 1, target:size(2) do
--      for b = 1, target:size(1) do
--        if opt.gpuid < 0 then
--          if predict:dim() == 1 then
--            if predict[b] == target[b][t] then
--              hit = hit + 1
--            end
--          else
--            if predict[b][t] == target[b][t] then
--              hit = hit + 1
--            end
--          end          
--        else
--          if predict[b][t][1] == target[b][t] then
--            hit = hit + 1
--          end
--        end
--        
--        ntotal = ntotal + 1
--      end
--    end
  end
  return loss / iter_per_epoch, hit / (B * T * iter_per_epoch)
end

-- TODO: something wrong of it
-- gradient check
--local nchecks = 10
--local eps = 1e-2
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
local hfolder = 'h' .. opt.rnn_size
if not path.exists(hfolder) then lfs.mkdir(hfolder) end

local val_loss_history = {}
local val_accuracy_history = {}
local iterations = {}
local epochs = {}
net:training()
local iter_per_epoch = loader.nbatch.train -- for loader4
local optim_opt = {learningRate = opt.learning_rate}
local num_iterations = iter_per_epoch * opt.nepochs
local checkpoint = {}
for i = 1, num_iterations do
--  local _, loss = optim.adagrad(feval, params, optim_opt)
  local epoch = math.ceil(i / iter_per_epoch)
  if i % iter_per_epoch == 0 then
    net:resetStates()
    
    -- Maybe decay learning rate
    if checkpoint.epoch % opt.lr_decay_every == 0 then
      local old_lr = optim_opt.learningRate
      optim_opt = {learningRate = old_lr * opt.lr_decay_factor}
    end
  end
  
  local _, loss = optim.adam(feval, params, optim_opt)
  
--  if i % iter_per_epoch == 0 then
  local check_every = opt.checkpoint_every
  if i % iter_per_epoch == 0 or (check_every > 0 and i % check_every == 0) or i == num_iterations then
    checkpoint.iter = i
    checkpoint.epoch = epoch
    checkpoint.model = net
    checkpoint.opt = opt
    checkpoint.loss = loss[1]
    net:evaluate()
--    net:resetStates()
--    checkpoint.train_accuracy = calAccuracy("train")
    net:resetStates()
    local val_loss, val_accuracy = calAccuracy("validate")
    checkpoint.validate_loss = val_loss
    checkpoint.validate_accuracy = val_accuracy
    print("i = ", i, " epoch = ", epoch, "val_loss = ", val_loss, "val_accuracy = ", val_accuracy)
    table.insert(val_loss_history, val_loss)
    table.insert(val_accuracy_history, val_accuracy)
    table.insert(iterations, i)
    table.insert(epochs, epoch)
    local save_every = opt.save_every
    if (i % iter_per_epoch == 0) or i == num_iterations then
      local savefile = string.format('%s/epoch%d_i%d.t7', hfolder, checkpoint.epoch, i)
      torch.save(savefile, checkpoint)
    end
    
    net:resetStates()
    net:training()
  end
  
--  xlua.progress(i,iter_per_epoch * opt.nepochs)
end

--local file = string.format('%s/stats_wordvec20.t7', hfolder)
--local stats = {}
--stats.opt = opt
--stats.iterations = iterations
--stats.epochs = epochs
--stats.val_loss_history = val_loss_history
--stats.val_accuracy_history = val_accuracy_history
--torch.save(file, stats)

