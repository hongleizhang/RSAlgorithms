import matplotlib.pylab as plt




def show_rmse():
	'''
	show figure 
	'''
	mfr=[0.85119,0.85401,0.86488,0.87981,0.90030] #PMF
	sr=[0.83730,0.82967,0.84495,0.85632,0.87138] #SocialReg
	my=[0.83607,0.82838,0.84288,0.85245,0.86950] #SocialEmbeddings
	x=[0.8,0.7,0.6,0.5,0.4]

	plt.plot(x,mfr,label='PMF')
	plt.plot(x,sr,label='SocialReg')
	plt.plot(x,my,linewidth='2',label='SocialEmbeddings')

	plt.xlabel('ratio of train set')
	plt.ylabel('RMSE')
	plt.title('FilmTrust(d=180)')
	plt.legend()
	plt.show()
	pass


show_rmse()