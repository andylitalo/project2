import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = np.loadtxt('data/data.txt').astype(int)
movies = np.genfromtxt('data/movies.txt', delimiter='\t', dtype=None,encoding=None, 
                       names=['Movie ID','Title','Unknown','Action','Adventure','Animation','Childrens','Comedy','Crime',
                             'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi',
                              'Thriller','War','Western'], deletechars='')

# Parse movie titles for later plotting
mtitles = np.copy(movies['Title'])
myears = np.copy(movies['Title'])
for j in np.arange(0,1682):
    # Strip quotes and year
    mtitles[j] = mtitles[j].replace('"', '')[:-7]
    # Make the titles more readable
    if mtitles[j][-5:] == ', The':
        mtitles[j] = 'The ' + mtitles[j][:-5]
    if mtitles[j][-3:] == ', A':
        mtitles[j] = 'A ' + mtitles[j][:-3]
    if mtitles[j][-4:] == ', La':
        mtitles[j] = 'La ' + mtitles[j][:-4]
    if mtitles[j][-4:] == ', Il':
        mtitles[j] = 'Il ' + mtitles[j][:-4]
        
    myears[j] = myears[j].replace('"', '')[-5:-1]

myears[266] = -1
myears[1127] = 1995
myears[1200] = 1996
myears[1411] = 1995
myears[1634] = 1986
myears = np.array([int(i) for i in myears])

# choice of optimizer: 'plain', 'bias' or 'oos'
optimization = 'bias'

Y_train = np.loadtxt('data/train.txt').astype(int)
Y_test = np.loadtxt('data/test.txt').astype(int)

M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
N = movies.shape[0] # movies
    
if optimization == 'plain':
    from collab_0 import train_model, get_err

    print('Factorizing with plain model.')

    reg = 0.1
    eta = 0.03 # learning rate
    K = 20

    print("Training model with M = %s, N = %s, k = %s, eta = %s, reg = %s"%(M, N, K, eta, reg))
    U, V, err = train_model(M, N, K, eta, reg, Y_train)
    V = V.T # todo: change in py file
    print('Factorization complete.')

elif optimization == 'bias':
    from collab_bias import train_model, get_err

    print('Factorizing with bias model.')

    reg = 0.1
    eta = 0.03 # learning rate
    K = 20

    print("Training model with M = %s, N = %s, k = %s, eta = %s, reg = %s"%(M, N, K, eta, reg))
    U, V, a, b, mu, err = train_model(M, N, K, eta, reg, Y_train, gbias=True)
    V = V.T # todo: change in py file
    print('Factorization complete.')
    
elif optimization == 'oos':
    # oos model using http://surprise.readthedocs.io/en/stable/matrix_factorization.html
    from surprise.prediction_algorithms.matrix_factorization import NMF
    from surprise import Dataset
    from surprise import Reader
    from surprise import accuracy
    
    reader = Reader(line_format='user item rating', sep='\t')
    model = NMF(n_factors=20)
    surprise_train = Dataset.load_from_file('data/data.txt', reader=reader)
    #surprise_test = Dataset.load_from_file('data/test.txt', reader=reader)
    model.fit(surprise_train.build_full_trainset())
    #predictions = model.test(surprise_test.build_full_trainset().build_testset())
    #err_test = accuracy.rmse(predictions)**2
    V = model.qi
    #print('Test error: %f' % err_test)
    
else:
    print('Invalid optimization method specified')
    
# number of movies to visualize
nmovies = 30

# opacity of axis in plots
axis_opacity = 0.3;

# Perform the SVD

# Following convention in the guide where V is KxN
V_centered = V-np.tile(np.resize(np.mean(V,axis=1),(K,1)),(1,V.shape[1]))
A,Si,B = np.linalg.svd(V_centered, full_matrices=False)
VT = np.dot(A[:,0:2].T,V_centered)

# Normalize data for the plots
VT[0] = (VT[0] - np.mean(VT[0]))/np.std(VT[0])
VT[1] = (VT[1] - np.mean(VT[1]))/np.std(VT[1])

# Find the data range for the plots
ylim = [-3,3]
xlim = [-3,3]
yrange = ylim[1]-ylim[0]
xrange = xlim[1]-xlim[0]

r_counts = np.bincount(data[:,1])[1:]
pop_inds_mov = np.argpartition(r_counts,-nmovies)[-nmovies:]
inds = np.asarray([49,257,99,180,293,285,287,0,299,120])

fig,ax = plt.subplots(figsize=(5,5))
ax.scatter(VT[0,inds],VT[1,inds])
plt.title('Principal components of the %i most popular movies' % (nmovies));
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.ylim(ylim)
plt.xlim(xlim)
plt.plot([xlim[0], xlim[1]], [0, 0], 'k--', lw=2, alpha=axis_opacity)
plt.plot([0, 0], [ylim[0], ylim[1]], 'k--', lw=2, alpha=axis_opacity)

## Label movies
for j in inds:
    ax.annotate(mtitles[j],(VT[0,j]+0.015*xrange,VT[1,j]-0.008*yrange),fontsize=14)