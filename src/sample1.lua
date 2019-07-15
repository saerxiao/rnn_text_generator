require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'model.LanguageModel'


local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'checkpoint/tinyshakespeare/lstm2/t64/h128/epoch40_i5560.t7') --epoch60_i8340.t7') --epoch1_i139.t7')        --t50/h128/epoch50.00.t7')
--cmd:option('-checkpoint', 'checkpoint/qts/lstm2/t64/h128/epoch45_i26595.t7') --h256/epoch20_i11820.t7, h128/epoch21_i12411.t7, 
--cmd:option('-checkpoint', 'checkpoint/qsc/lstm2/t64/h128/epoch60_i8160.t7') -- epoch30_i4080.t7 -- epoch60_i8160.t7
cmd:option('-length', 3000)
cmd:option('-start_text', '')
cmd:option('-sample', 1)
cmd:option('-temperature', 1)
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-verbose', 0)
local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

--local msg
--if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
--  require 'cutorch'
--  require 'cunn'
--  cutorch.setDevice(opt.gpu + 1)
--  model:cuda()
--  msg = string.format('Running with CUDA on GPU %d', opt.gpu)
--elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
--  require 'cltorch'
--  require 'clnn'
--  model:cl()
--  msg = string.format('Running with OpenCL on GPU %d', opt.gpu)
--else
--  msg = 'Running in CPU mode'
--end
--if opt.verbose == 1 then print(msg) end

model:evaluate()

local sample = model:sample(opt)
print(sample)
