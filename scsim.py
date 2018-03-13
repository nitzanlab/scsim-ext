import pandas as pd
import numpy as np

class scsim:
    def __init__(self, ngenes=10000, ncells=100, seed=757578,
                 mean_rate=.3, mean_shape=.6, libloc=11, libscale=0.2,
                expoutprob=.05, expoutloc=4, expoutscale=0.5, ngroups=1,
                diffexpprob=.1, diffexpdownprob=.5,
                diffexploc=.1, diffexpscale=.4, bcv_dispersion=.1,
                bcv_dof=60, ndoublets=0, groupprob=None,
                nproggenes=None, progdownprob=None, progdeloc=None,
                progdescale=None, proggoups=None, progcellfrac=None,
                minprogusage=.2, maxprogusage=.8):

        self.ngenes = ngenes
        self.ncells = ncells
        self.seed = seed
        self.mean_rate = mean_rate
        self.mean_shape = mean_shape
        self.libloc = libloc
        self.libscale = libscale
        self.expoutprob = expoutprob
        self.expoutloc = expoutloc
        self.expoutscale = expoutscale
        self.ngroups = ngroups
        self.diffexpprob = diffexpprob
        self.diffexpdownprob = diffexpdownprob
        self.diffexploc = diffexploc
        self.diffexpscale = diffexpscale
        self.bcv_dispersion = bcv_dispersion
        self.bcv_dof = bcv_dof
        self.ndoublets = ndoublets
        self.init_ncells = ncells+ndoublets
        self.nproggenes=nproggenes
        self.progdownprob=progdownprob
        self.progdeloc=progdeloc
        self.progdescale=progdescale
        self.proggoups=proggoups
        self.progcellfrac = progcellfrac
        self.minprogusage = minprogusage
        self.maxprogusage = maxprogusage

        if groupprob is None:
            self.groupprob = [1/float(self.ngroups)]*self.ngroups
        elif (len(groupprob) == self.ngroups) & (np.sum(groupprob) == 1):
            self.groupprob = groupprob
        else:
            sys.exit('Invalid groupprob input')


    def simulate(self):
        print('Simulating cells')
        self.cellparams = self.get_cell_params()
        print('Simulating gene params')
        self.geneparams = self.get_gene_params()

        if (self.nproggenes is not None) and (self.nproggenes > 0):
            print('Simulating program')
            self.simulate_program()

        print('Simulating DE')
        self.sim_group_DE()
        print('Simulating cell-gene means')
        self.cellgenemean = self.get_cell_gene_means()
        if self.ndoublets > 0:
            print('Simulating doublets')
            self.simulate_doublets()

        print('Adjusting means')
        self.adjust_means_bcv()
        print('Simulating counts')
        self.simulate_counts()

    def simulate_counts(self):
        '''Sample read counts for each gene x cell from Poisson distribution
        using the variance-trend adjusted updatedmean value'''
        self.counts = pd.DataFrame(np.random.poisson(lam=self.updatedmean),
                                   index=self.cellnames, columns=self.genenames)

    def adjust_means_bcv(self):
        '''Adjust cellgenemean to follow a mean-variance trend relationship'''
        self.bcv = (self.bcv_dispersion + (1 / np.sqrt(self.cellgenemean.astype(float))))
        chisamp = np.random.chisquare(self.bcv_dof, size=self.ngenes)
        self.bcv = self.bcv*np.sqrt(self.bcv_dof / chisamp)
        self.updatedmean = np.random.gamma(shape=1/(self.bcv**2),
                                           scale=self.cellgenemean*(self.bcv**2))
        self.bcv = pd.DataFrame(self.bcv, index=self.cellnames, columns=self.genenames)
        self.updatedmean = pd.DataFrame(self.updatedmean, index=self.cellnames,
                                        columns=self.genenames)


    def simulate_doublets(self):
        ## Select doublet cells and determine the second cell to merge with
        d_ind = sorted(np.random.choice(self.ncells, self.ndoublets,
                                        replace=False))
        d_ind = ['Cell%d' % (x+1) for x in d_ind]
        self.cellparams['is_doublet'] = False
        self.cellparams.loc[d_ind, 'is_doublet'] = True
        extraind = self.cellparams.index[-self.ndoublets:]
        group2 = self.cellparams.ix[extraind, 'group'].values
        self.cellparams['group2'] = -1
        self.cellparams.loc[d_ind, 'group2'] = group2

        ## update the cell-gene means for the doublets while preserving the
        ## same library size
        dmean = self.cellgenemean.loc[d_ind,:]
        dmean = dmean.multiply(.5 / dmean.sum(axis=1), axis=0)
        omean = self.cellgenemean.loc[extraind,:]
        omean = omean.multiply(.5 / omean.sum(axis=1), axis=0)
        newmean = dmean + omean.values
        newmean = newmean.multiply(self.cellparams.loc[d_ind, 'libsize'], axis=0)
        self.cellgenemean.loc[d_ind,:] = newmean
        ## remove extra doublet cells from the data structures
        self.cellgenemean.drop(extraind, axis=0, inplace=True)
        self.cellparams.drop(extraind, axis=0, inplace=True)
        self.cellnames = self.cellnames[0:self.ncells]


    def get_cell_gene_means(self):
        '''Calculate each gene's mean expression for each cell by adjusting
        for the cells library size'''
        ind = self.cellparams['group'].apply(lambda x: 'group%d_genemean' % x)

        if self.nproggenes == 0:
            cellgenemean = self.geneparams.loc[:,ind].T
            cellgenemean.index = self.cellparams.index
        else:
            print('   - Getting group means')
            groupmean = self.geneparams.loc[:,ind].T
            groupmean.index = self.cellparams.index
            print('   - Normalizing group means')
            groupmean = groupmean.div(groupmean.sum(axis=1), axis=0)
            ncells = len(ind)
            print('   - Getting program means')
            progmean = self.geneparams.loc[:,['prog_genemean']*ncells].T
            progmean.index = self.cellparams.index
            print('   - Normalizing program means')
            progmean = progmean.div(progmean.sum(axis=1), axis=0)
            cellgenemean = progmean.multiply(self.cellparams['program_usage'], axis=0)
            del(progmean)
            cellgenemean = cellgenemean + groupmean.multiply(1 - self.cellparams['program_usage'], axis=0)
            del(groupmean)

        print('   - Normalizing by cell libsize')
        normfac = self.cellparams['libsize'] / cellgenemean.sum(axis=1)
        cellgenemean = cellgenemean.multiply(normfac, axis=0).astype(float)
        return(cellgenemean)


    def get_gene_params(self):
        '''Sample each genes mean expression from a gamma distribution as
        well as identifying outlier genes with expression drawn from a
        log-normal distribution'''
        basegenemean = np.random.gamma(shape=self.mean_shape,
                                       scale=1./self.mean_rate,
                                       size=self.ngenes)

        is_outlier = np.random.choice([True, False], size=self.ngenes,
                                      p=[self.expoutprob,1-self.expoutprob])
        outlier_ratio = np.ones(shape=self.ngenes)
        outliers = np.random.lognormal(mean=self.expoutloc,
                                       sigma=self.expoutscale,
                                       size=is_outlier.sum())
        outlier_ratio[is_outlier] = outliers
        gene_mean = basegenemean.copy()
        median = np.median(basegenemean)
        gene_mean[is_outlier] = outliers*median
        self.genenames = ['Gene%d' % i for i in range(1, self.ngenes+1)]
        geneparams = pd.DataFrame([basegenemean, is_outlier, outlier_ratio, gene_mean],
                                  index=['BaseGeneMean', 'is_outlier', 'outlier_ratio', 'gene_mean'],
                                 columns=self.genenames).T
        return(geneparams)


    def get_cell_params(self):
        '''Sample cell group identities and library sizes'''
        groupid = self.simulate_groups()
        libsize = np.random.lognormal(mean=self.libloc, sigma=self.libscale,
                                      size=self.init_ncells)
        self.cellnames = ['Cell%d' % i for i in range(1, self.init_ncells+1)]
        cellparams = pd.DataFrame([groupid, libsize],
                                  index=['group', 'libsize'],
                                  columns=self.cellnames).T
        cellparams['group'] = cellparams['group'].astype(int)
        return(cellparams)

    def simulate_program(self):
        ## Simulate the program gene expression
        self.geneparams['prog_gene'] = False
        proggenes = self.geneparams.index[-self.nproggenes:]
        self.geneparams.loc[proggenes, 'prog_gene'] = True
        DEratio = np.random.lognormal(mean=self.progdeloc,
                                          sigma=self.progdescale,
                                          size=self.nproggenes)
        DEratio[DEratio<1] = 1 / DEratio[DEratio<1]
        is_downregulated = np.random.choice([True, False],
                                            size=len(DEratio),
                                            p=[self.progdownprob,
                                            1-self.progdownprob])
        DEratio[is_downregulated] = 1. / DEratio[is_downregulated]
        all_DE_ratio = np.ones(self.ngenes)
        all_DE_ratio[-self.nproggenes:] = DEratio
        prog_mean = self.geneparams['gene_mean']*all_DE_ratio
        self.geneparams['prog_genemean'] = prog_mean

        ## Assign the program to cells
        self.cellparams['has_program'] = False
        if self.proggoups is None:
            ## The program is active in all cell types
            self.proggoups = np.arange(1, self.ngroups+1)

        self.cellparams.loc[:, 'program_usage'] = 0
        for g in self.proggoups:
            groupcells = self.cellparams.index[self.cellparams['group']==g]
            hasprog = np.random.choice([True, False], size=len(groupcells),
                                      p=[self.progcellfrac,
                                      1-self.progcellfrac])
            self.cellparams.loc[groupcells[hasprog], 'has_program'] = True
            usages = np.random.uniform(low=self.minprogusage,
                                       high=self.maxprogusage,
                                       size=len(groupcells[hasprog]))
            self.cellparams.loc[groupcells[hasprog], 'program_usage'] = usages




    def simulate_groups(self):
        '''Sample cell group identities from a categorical distriubtion'''
        groupid = np.random.choice(np.arange(1, self.ngroups+1),
                                       size=self.init_ncells, p=self.groupprob)
        self.groups = np.unique(groupid)
        return(groupid)


    def sim_group_DE(self):
        '''Sample differentially expressed genes and the DE factor for each
        cell-type group'''
        groups = self.cellparams['group'].unique()
        if self.nproggenes>0:
            proggene = self.geneparams['prog_gene'].values
        else:
            proggene = np.array([False]*self.geneparams.shape[0])

        for group in self.groups:
            isDE = np.random.choice([True, False], size=self.ngenes,
                                      p=[self.diffexpprob,1-self.diffexpprob])
            isDE[proggene] = False # Program genes shouldn't be differentially expressed between groups
            DEratio = np.random.lognormal(mean=self.diffexploc,
                                          sigma=self.diffexpscale,
                                          size=isDE.sum())
            DEratio[DEratio<1] = 1 / DEratio[DEratio<1]
            is_downregulated = np.random.choice([True, False],
                                            size=len(DEratio),
                                            p=[self.diffexpdownprob,1-self.diffexpdownprob])
            DEratio[is_downregulated] = 1. / DEratio[is_downregulated]
            all_DE_ratio = np.ones(self.ngenes)
            all_DE_ratio[isDE] = DEratio
            group_mean = self.geneparams['gene_mean']*all_DE_ratio

            deratiocol = 'group%d_DEratio' % group
            groupmeancol = 'group%d_genemean' % group
            self.geneparams[deratiocol] = all_DE_ratio
            self.geneparams[groupmeancol] = group_mean
