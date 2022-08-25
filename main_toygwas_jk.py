"""
Last updated on 2022-05-13
@author1: Shu Zixin (zixshu@student.ethz.ch)
@author2: Juliane Klatt (juliane.klatt@bsse.ethz.ch)

Toy model for gene representation learning
"""
### PACKAGES ##################################################################

from __future__           import print_function
import argparse
import numpy              as np
import torch
import torch.optim        as optim
from   model_gwas         import Attention,GatedAttention
from   torch.autograd     import Variable
from   torch.utils.data   import TensorDataset,DataLoader

### GLOBAL VARIABLES ##########################################################

# argument parsing
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')

parser.add_argument('--epochs',
                    type=int,
                    default=20,
                    metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr',
                    type=float, 
                    default=0.0005, 
                    metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', 
                    type=float, 
                    default=10e-5, 
                    metavar='R',
                    help='weight decay')
parser.add_argument('--mean_bag_length',
                    type=int, 
                    default=10, 
                    metavar='ML',
                    help='average number of SNPs present')
parser.add_argument('--var_bag_length',
                    type=int, 
                    default=2, 
                    metavar='VL',
                    help='variance of number of SNPs present')
parser.add_argument('--num_bags_train',
                    type=int,
                    default=800,
                    metavar='NTrain',
                    help='number of samples in training set')
parser.add_argument('--num_bags_test', 
                    type=int, 
                    default=200, 
                    metavar='NTest',
                    help='number of samples in test set')
parser.add_argument('--seed', 
                    type=int, 
                    default=42, 
                    metavar='S',
                    help='random seed')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--model',
                    type=str,
                    default='attention',
                    help='Choose b/w attention and gated_attention')

args      = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')
    
### MAIN ######################################################################

def main():
    
    # define number of snps on the gene of interest, maximum number of snps
    # present in a sample, and identifiers of causal snps
    n_gsnps = 100
    m_psnps = 20
    csnps   = [1,2]
    
    # generate genotype, snp labels and phenotypes
    n_smpls  = args.num_bags_train+args.num_bags_test
    gs,ls,ps = generate_samples(n_gsnps,m_psnps,csnps,n_smpls)
    
    # set up test and training split
    train_data,train_loader,test_data,test_loader = train_test_split(gs,ps,ls)
    
    # initialize the model of choice
    model = init_model()
    
    # setup optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           betas=(0.9, 0.999),
                           weight_decay=args.reg)
    
    # train
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch,train_loader,model,optimizer)
        
    # test
    print('Start Testing')
    test(test_loader,model)
    
    return None

### FUNCTIONS #################################################################

def init_model():
    
    print('Initialize Model...')
    if args.model=='attention':
        model = Attention()
    elif args.model=='gated_attention':
        model = GatedAttention()
    if args.cuda:
        model.cuda()
    
    return model

def train_test_split(gs,ps,ls):
    
    print('Generate Train and Test Set...')
    
    # define train and test set sizes
    n_train = args.num_bags_train
    
    # split into train and test set
    train_data   = TensorDataset(torch.tensor(gs[:n_train]),
                                 torch.tensor(ps[:n_train]),
                                 torch.tensor(ls[:n_train]))
    train_loader = DataLoader(train_data,batch_size=1,shuffle=False)
    test_data    = TensorDataset(torch.tensor(gs[n_train:]),
                                 torch.tensor(ps[n_train:]),
                                 torch.tensor(ls[n_train:]))
    test_loader  = DataLoader(test_data,batch_size=1,shuffle=False)
    
    return train_data,train_loader,test_data,test_loader

def generate_samples(n_gsnps,m_psnps,csnps,n_smpls):
    """Generates some toy samples of GWAS data.

    Parameters
    ----------
    n_gsnps: `int`
        number of snps located on the gene of interest
    m_psnps: `int`
        maximum number of snps present in a sample
    csnps: `list`
        identifiers of snps needed to cause the phenotype
    n_smpls: `int`
        number of samples
        
    Returns
    -------
    genotype: `list`
        list of n_smpls lists of present snps
    snp_labels: `list`
        list of n_smpls lists of snp labels indicating whether or not a present
        snp is causal
    phenotype: `list`
        list of n_smpls phenotypes
    """

    # generate random number of present snps for each sample (w/ replacement)
    n_psnps = np.random.randint(low=1,high=m_psnps+1,size=n_smpls)
    
    # allocate genotypes and phenotypes
    genotypes  = []
    snp_labels = []
    phenotypes = []
    
    # create sample genotypes and phenotypes
    for s in range(n_smpls):
        
        # create genotype w/ number of present snps in that sample
        # beware: random drawing of snp identifiers must be w/o replacement!
        # therefore np.random.choice(), instead of np.random.randiint()
        genotype = np.random.choice(np.arange(1,n_gsnps+1),
                                    size=n_psnps[s],
                                    replace=False).tolist()
        
        # pad genotype with zeros, such that all genotypes have length m_psnps
        # beware: only makes sense if '0' is not among the snp identifiers!
        genotype += [0]*(m_psnps-len(genotype))
        
        # make each entry in the genotype a list, such that it carries snp
        # properties in addition to the snp identifier
        genotype = [[[s,1,1]] for s in genotype]
        # TODO: add real properties that are looked up in a snp-dictionary
        
        # assign snp labels: 1 if causal, 0 if not causal
        snp_label = [int(snp[0][0] in csnps) for snp in genotype]
        
        # compute phenotype: 1 if at least one causal snp present, 0 if not
        phenotype = [int(sum(snp_label)>0)]
        # TODO: implement logical AND (i.e., interaction), right now its XOR
        
        # append
        genotypes  += [genotype]
        snp_labels += [snp_label]
        phenotypes += [phenotype]
    
    return genotypes,snp_labels,phenotypes

def train(epoch,train_loader,model,optimizer):
    
    model.train()
    train_loss = 0.
    train_error = 0.
    
    for batch_idx, (data,bag_label,label) in enumerate(train_loader):
        
        # bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
            
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        
        # calculate loss and metrics
        loss, _      = model.calculate_objective(data,bag_label)
        train_loss  += loss.data[0]
        error, _     = model.calculate_classification_error(data, bag_label)
        train_error += error
        
        # backward pass
        loss.backward()
        
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
 
    train_log(epoch,train_loss,train_error)
    
    return None

def train_log(epoch,loss,error):
    
    e = epoch
    l = loss.cpu().numpy()[0]
    r = error
    
    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(e,l,r))
    
    return None

def test(test_loader,model):
    
    model.eval()
    test_loss = 0.
    test_error = 0.
    
    for batch_idx, (data,bag_label,label) in enumerate(test_loader):
        
        instance_labels = label
        
        if args.cuda:
            data,bag_label = data.cuda(),bag_label.cuda()
            
        data,bag_label   = Variable(data),Variable(bag_label)
        loss,attention   = model.calculate_objective(data,bag_label)
        test_loss       += loss.data[0]
        error,pred_label = model.calculate_classification_error(data,bag_label)
        test_error      += error

        # print bag labels and instance labels for minority phenotype
        bag_level      = (bag_label.cpu().data.numpy()[0],
                          int(pred_label.cpu().data.numpy()[0][0]))
        inst_att       = attention.cpu().data.numpy()[0]
        inst_att       = [np.round(a,3) for a in inst_att]
        inst_label     = instance_labels.numpy()[0].tolist()
        instance_level = list(zip(inst_label,inst_att))
        
        if int(pred_label.cpu().data.numpy()[0][0])==1:

            pred_log(bag_level,instance_level)

    test_error /= len(test_loader)
    test_loss  /= len(test_loader)

    test_log(test_loss,test_error)

    return None

def pred_log(bags,instances):
    
    instances = [i for i in instances if i[1]!=0]
    
    print('\nTrue phenotype:      {}'.format(bags[0][0]))
    print(  'Predicted pehnotype: {}'.format(bags[1]))
    print('\nSNP labels, attention weights:\n{}'.format(instances))
    
    return None

def test_log(loss,error):
    
    l = loss.cpu().numpy()[0]
    e = error
    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(l,e))
    
    return None

###############################################################################

if __name__ == "__main__":
    
    main()
