require('nn')
require('cltorch')
require('clnn')
require('optim')


local opt = lapp[[
   -b,--batchSize     (default 100)          batch size
   --backend (string default clnn)       clnn for openCL or nn for CPU
   -e,--epochs (default 10)                 how many epochs it should do
]]

-- os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
-- os.execute('unzip cifar10torchsmall.zip')
torch.setnumthreads(4)

print(opt.backend)

if opt.backend ~= 'nn' and opt.backend ~= 'clnn' then
    error('backend should be nn or clnn')
end

geometry = {32, 32}

classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

confusion = optim.ConfusionMatrix(classes)

torch.manualSeed(1)

cltorch.synchronize()

net = nn.Sequential()

net:add(nn.SpatialConvolutionMM(3, 6, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolutionMM(6, 16, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))
net:add(nn.Linear(16*5*5, 120))
net:add(nn.Linear(120, 84))
net:add(nn.Linear(84, 10))
net:add(nn.LogSoftMax())
--net:add(nn.Reshape(1, 10))

if backend == 'clnn' then
    net = net:cl()
end
parameters,gradParameters = net:getParameters()


trainset = torch.load('cifar10-train.t7')
trainset.data = trainset.data:double()

testset = torch.load('cifar10-test.t7')
testset.data = testset.data:double()


mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

if backend == 'clnn' then
    trainset.data = trainset.data:cl()
end

setmetatable(trainset, 
    {__index = function(t, i) 
		return {t.data[i], t.label[i]} 
	end}
);

function trainset:size() 
    return self.data:size(1) 
end

if backend == 'clnn' then
    trainset.label = trainset.label:double()
    trainset.label = trainset.label:cl()
end

print(trainset)
--print(net:forward(trainset.data[4]))

    criterion = nn.ClassNLLCriterion() -- a negative log-likelihood criterion for multi-class classification
if backend == 'clnn' then
    criterion = criterion:cl()
end


function train(dataset)
    epoch = epoch or 1

    local time = sys.clock()

    for t = 1,dataset:size(),opt.batchSize do
        local curBatchSize = math.min(opt.batchSize,dataset:size()-t)

        local inputs = torch.Tensor(curBatchSize,3,geometry[1],geometry[2])
        local targets = torch.Tensor(curBatchSize)

        if backend == 'clnn' then
            inputs = inputs:cl()
            targets = targets:cl()
        end

        local k = 1

        for i = t, t+curBatchSize-1 do
            local sample = dataset[i]
            
            local input = sample[1]:clone()
            local target = sample[2]
            --target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end

        -- create closure to evaluate f(X) and df/dX

        local feval = function(x)
            -- just in case:
            collectgarbage()

            if x ~= parameters then
                parameters:copy(x)
            end

            gradParameters:zero()

            local outputs = net:forward(inputs)

            local f = criterion:forward(outputs, targets)

             -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            net:backward(inputs, df_do)

            for i = 1,curBatchSize do
                confusion:add(outputs[i], targets[i])
            end

            -- return f and df/dX
            return f,gradParameters
        end

        sgdState = sgdState or {
            learningRate = 0.005,
            momentum = 0,
            learningRateDecay = 0
        }
        
        optim.sgd(feval, parameters, sgdState)

        xlua.progress(t+curBatchSize-1, dataset:size())
    end

    print(confusion)

    confusion:zero()

    epoch = epoch + 1
end

local totalTime = sys.clock()

for i=1,opt.epochs do
    train(trainset)
end

totalTime = sys.clock() - totalTime


testset.data = testset.data:double()
net = net:double()
correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i]):reshape(10)
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print('label is ', groundtruth, ' prediction is ', indices[1], ' state is ', prediction)
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print('totalTime to train was ', totalTime, ' and got ', correct, 100*correct/10000 .. ' % on test')