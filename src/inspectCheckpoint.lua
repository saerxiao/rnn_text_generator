require 'lfs'
require 'torch'
require 'gnuplot'
require 'nn'
require 'cunn'
require 'cutorch'
require 'model.MultiLayerRnn1'
require 'model.MultiLayerRnn2'
require 'model.LanguageModel'
--require 'model.VanillaRnn'

function plot_accuracy(dir)
  local current = lfs.currentdir()
  dir = current .. "/" .. dir
  lfs.chdir(dir)
  print(lfs.currentdir())
--  local dir = 'checkpoint/'
  local files = {}
  for file in lfs.dir(dir) do
    if lfs.attributes(dir .. "/" .. file, "mode") == "file" then
      table.insert(files,dir .. "/" .. file)
    end
  end

  -- sort files by epoch
  local function find_iter(s)
    local epochStartIndex = string.find(s,"epoch")
    return tonumber(string.sub(s, epochStartIndex + string.len("epoch"), string.find(s, '%.')-1))
  end

  table.sort(files, function (a,b) return find_iter(a) < find_iter(b) end)

  local epochs = torch.Tensor(#files)
  local losses = torch.Tensor(#files)
  local train_accuracy = torch.Tensor(#files)
  local validate_accuracy = torch.Tensor(#files)
  for i = 1, #files do
    local f = torch.load(files[i])
    epochs[i] = f.epoch
    losses[i] = f.loss
    train_accuracy[i] = f.train_accuracy
    validate_accuracy[i] = f.validate_accuracy
    print(i, "loss = ", losses[i], "val_accuracy = ", validate_accuracy[i])
  end
  
  -- make plots
  gnuplot.raw('set multiplot layout 1,2')
  gnuplot.plot(epochs, losses, '+-')
  gnuplot.plot({epochs, train_accuracy, '+'}, {epochs, validate_accuracy, '+'})
end