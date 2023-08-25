import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union


if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [0] * 10
        self.L = 0.0
    
    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.

        Returns:
            np.array: Predicted score possible
        """
        pred = Z_0[w-1] * (1 - np.exp(-L*X/Z_0[w-1]))
        return pred
        

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters 
                   over the given datapoints.
        """
        
        loss=0
        Z=Params[:-1]
        L=Params[-1]
        for j in range(len(X)):
            y_pred = self.get_predictions(X,Z,w,L)
            loss+= (Y[j] - y_pred[j])**2 
        
        return loss
        
    
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    df = pd.read_csv(data_path)
    return df

def correct_date_format(dt: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Check date column and convert to dd/mm/yyyy format
    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    month_index = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    print("Date format corrected.")
    for i in dt.index:
        date = str(dt['Date'][i])
        if len(date)>10:
            strs = date.split(" ")
            day = strs[1].split("-")[0]
            if len(day) == 1:
                day = "0" + day
            month = month_index[strs[0]]
            year = strs[2]
            dt.loc[:,('Date','i')] =  str(day) + "/" + str(month) + "/" + str(year)
    
    return dt
    

def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    #correct the format of date in 'Date' column
    data=correct_date_format(data)
    
    # add Over.Remaining column in data and pick data with 1st innings only
   
    data=data[data['Innings'] == 1] #taking only 1st innings
    data=data[data['Error.In.Data']==0] #taking data with zero error
    data['Over.Remaining']=50-data['Over']
    
    #keep data that start with over=1
    x = data.groupby(['Match'],as_index=False)['Over'].min()
    x=x[x['Over']==1]
    data=pd.merge(data,x,on='Match',how='inner')
    
    #following columns only required to test and train D/L model
    data=data[['Over.Remaining','Runs.Remaining','Wickets.in.Hand']]

    return data


def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """
    #parameters to pass to optimize function
    PARAMS=model.Z0
    PARAMS.append(model.L)
    
    print("Model Trainig start")
    for w in range(1,11):
    
        df=data[data['Wickets.in.Hand']==w] #taking data for particular wicket
        
        #### use it when error calculation will use average Runs remaining for each over ###
        df = df.groupby(["Over.Remaining"],as_index=False)["Runs.Remaining"].mean()
        
        overs = df["Over.Remaining"].to_numpy()
        label = df["Runs.Remaining"].to_numpy()
    
        PARAMS[w-1] = data["Runs.Remaining"].mean() #taking inital value for optimizer
        
        opt = sp.optimize.minimize(model.calculate_loss, x0=PARAMS ,args=(overs,label,w) ,method='Powell')
        
        model.Z0[w-1]=opt.x[w-1]
        model.L=opt.x[-1]
        
    model.Z0.pop() #z0 size increased, need to decrease

    print("Training finished")
    return model

def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    fig = plt.subplots(figsize=(8,5))
    for i in range(9,-1,-1):
        x = np.linspace(0,60,61)
        pred_y = model.get_predictions(x, model.Z0, i+1, model.L)
        plt.plot(x,pred_y)

    plt.ylabel("Average runs scorable")
    plt.xlabel("Overs remainig")
    plt.xlim([0,50])
    plt.ylim([0,270])
    plt.title('D/L method')
    plt.legend(['10','9','8','7','6','5','4','3','2','1'],title='Wickets')
    plt.grid(True,linestyle=':',color='r', alpha=0.5)
    plt.savefig(plot_path)
    plt.show()
    


def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''

    print("Z_0 = ",model.Z0)
    print("L = ",model.L)
    parameters = model.Z0
    parameters.append(model.L)
    return parameters


def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    mse = 0.0
    length=0
    params=model.Z0
    params.append(model.L)
    for w in range(1,11):
        
        df=data[data['Wickets.in.Hand']==w]
        
        #### use it when error calculation will use average Runs remaining ###
        #df = df.groupby(["Over.Remaining"],as_index=False)["Runs.Remaining"].mean() 
        
        overs = df["Over.Remaining"].to_numpy()
        labels = df["Runs.Remaining"].to_numpy()
        mse+=model.calculate_loss(params,overs,labels,w)
        length+=len(overs)
    mse/=length
    print('MSE = ',mse)
    return mse


def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.")
    
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    
    #model.load(args['model_path'])
    
    plot(model, args['plot_path'])  # Plotting the model
    
    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data)


if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)
