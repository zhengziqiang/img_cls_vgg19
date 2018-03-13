{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import random\n",
    "train_list=set(range(1,836))\n",
    "val=set(random.sample(range(1,836),int(836/5)))\n",
    "train_list=list(train_list-val)\n",
    "val=list(val)\n",
    "data=np.genfromtxt(\"C:\\\\Users\\\\Administrator\\\\Desktop\\\\数学建模\\\\项目一汇总.csv\",dtype=float,delimiter=',')\n",
    "train_data=np.zeros((668,13))\n",
    "val_data=np.zeros((167,13))\n",
    "for i in range(len(train_list)):\n",
    "    train_data[i]=(data[train_list[i]-1,3:])\n",
    "for i in range(len(val)):\n",
    "    val_data[i]=(data[val[i]-1,3:])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ -3.49326960e+01  -6.09441972e+00   6.55000000e+01   3.06789000e-01\n   5.66943000e-02   6.70000000e+01   3.95336000e+03   2.28782000e+08\n   2.00000000e+00   5.30612000e+01   4.39236000e+01   1.75880000e+01\n   0.00000000e+00]\n"
     ]
    }
   ],
   "source": [
    "print (train_data[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(668, 12)\n"
     ]
    }
   ],
   "source": [
    "no_price1=train_data[:,:2]\n",
    "no_price2=train_data[:,3:]\n",
    "no_price=np.concatenate([no_price1,no_price2],axis=1)\n",
    "print (no_price.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(167, 12)\n"
     ]
    }
   ],
   "source": [
    "no_price1_val=val_data[:,:2]\n",
    "no_price2_val=val_data[:,3:]\n",
    "no_price_val=np.concatenate([no_price1_val,no_price2_val],axis=1)\n",
    "print (no_price_val.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "val_=no_price_val[:,:11]\n",
    "price_val=val_data[:,2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(167, 11)\n"
     ]
    }
   ],
   "source": [
    "print (val_.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "def normalize(x):\n",
    "    min_x=np.min(x)\n",
    "    max_x=np.max(x)\n",
    "    for i in range(len(x)):\n",
    "        x[i]=(x[i]-min_x)/(max_x-min_x)\n",
    "    return x\n",
    "\n",
    "for i in range(11):\n",
    "    val_[:,i]=normalize(val_[:,i])\n",
    "price_val=normalize(price_val)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "price_val=np.reshape(price_val,[167,1])\n",
    "val_all=np.concatenate([val_,price_val],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "out=open(\"C:\\\\Users\\\\Administrator\\\\Desktop\\\\数学建模\\\\mlp_regression_val.csv\",\"w\")\n",
    "for i in range(np.shape(val_all)[0]):\n",
    "    for j in range(np.shape(val_all)[1]-1):\n",
    "        out.write(str(val_all[i][j])+',')\n",
    "    out.write(str(val_all[i][-1])+'\\n')\n",
    "out.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(668, 11)\n"
     ]
    }
   ],
   "source": [
    "print (train_.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def normalize(x):\n",
    "    min_x=np.min(x)\n",
    "    max_x=np.max(x)\n",
    "    for i in range(len(x)):\n",
    "        x[i]=(x[i]-min_x)/(max_x-min_x)\n",
    "    return x\n",
    "\n",
    "for i in range(11):\n",
    "    train_[:,i]=normalize(train_[:,i])\n",
    "price=normalize(price)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(668, 11)\n"
     ]
    }
   ],
   "source": [
    "print(np.shape(train_))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(668,)\n"
     ]
    }
   ],
   "source": [
    "print(np.shape(price))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "price=np.reshape(price,[668,1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_all=np.concatenate([train_,price],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "out=open(\"C:\\\\Users\\\\Administrator\\\\Desktop\\\\数学建模\\\\mlp_regression_train.csv\",\"w\")\n",
    "for i in range(np.shape(train_all)[0]):\n",
    "    for j in range(np.shape(train_all)[1]-1):\n",
    "        out.write(str(train_all[i][j])+',')\n",
    "    out.write(str(train_all[i][-1])+'\\n')\n",
    "out.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(668, 12)\n"
     ]
    }
   ],
   "source": [
    "print(np.shape(train_all))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    ""
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2.0
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}