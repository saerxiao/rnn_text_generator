require 'torch'
require 'lfs'

local files = {}
local dir = "data/qsc" -- data/qts, data/qsc
for file in lfs.dir(dir) do
  if lfs.attributes(dir .. "/" .. file, "mode") == "file" then
--    print(file)
    table.insert(files,dir .. "/" .. file)
  end
end

table.sort(files, function (a,b) return a < b end)
--print(files)

local function addToTable(vocab, t, a)
  if not vocab[a] then
    table.insert(vocab,a)
    vocab[a] = #vocab
  end
  table.insert(t, vocab[a])
end

local maxLoad = 2000000
local BUFSIZE = 50000
local vocab = {}
local inputT = {}
local prefix = {}
local charCnt = 0
table.insert(prefix, 3, tonumber("110",2))
table.insert(prefix, 4, tonumber("1110", 2))
table.insert(prefix,5, tonumber("11110", 2))


--local files = {"data/qts/qts_0001.txt", "data/qts/qts_0002.txt", "data/qts/qts_0003.txt"}

for _,filename in pairs(files) do
--local f = assert(io.input("data/chtest/input.txt"))
local f = assert(io.input(filename))
--local txt = f:read("*all")
--print(txt:len())
--local input = "秦川,秦川!"
local cc = 0
local input = ''
while true do
  local chunk = f:read(BUFSIZE)
  if not chunk then break end
  cc = cc + chunk:len()
  input = input .. chunk
  if maxLoad > 0 and cc > maxLoad then break end
end
--print(input)
--if not prev then
--  addToTable(t, prev .. input:sub(1, need))
--  prev, need = nil, nil
--  i = need + 1
--end
local i = 1
while i <= input:len() do
  
  if bit.rshift(input:byte(i,i), 7) == 0 then 
    addToTable(vocab, inputT, input:sub(i,i))
    i = i + 1
  else
    local found = false
    for l = 3,5 do
      if bit.rshift(input:byte(i,i), 8-l) == prefix[l] then
        charCnt = charCnt + 1
        if i + l - 2 < input:len() then
          addToTable(vocab, inputT, input:sub(i, i+l-2))
        end
        i = i + l -1
        found = true
        break
      end
    end
    if not found then
      error("cannot find matching character")
    end
  end
end

end -- end files for loop
--print(t)
local inputV = torch.Tensor(inputT)
--print(inputV[{{3000, 3020}}])
print(#vocab)
print(charCnt)
torch.save(dir .. "/vocab.t7", vocab)
torch.save(dir .. "/input_v.t7", inputV)