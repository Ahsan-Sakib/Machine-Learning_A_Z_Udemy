# Machine-Learning
Udemy machine learning course 

## Section 1

Data : https://www.superdatascience.com/pages/machine-learning


## Section 2

Feature scaling  : Column wise

## Section 3 : Data preprocessing

reading reference : https://www.geeksforgeeks.org/data-preprocessing-machine-learning-python/

    Step 1: Import the necessary libraries
    
    Step 2: Load the dataset
    
    Step 3: Statistical Analysis
    
    Step 4: Check the outliers:
    
    Step 5: Correlation
    
    Step 6: Separate independent features and Target Variables


## Section 4 : Simple Linear Regression 

        # import the library
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression


        # import the dataset
        
        df = pd.read_csv("Salary_Data.csv")


        x = df.iloc[:,:-1].values
        y = df.iloc[:,1].values

        # Train and Test dataset split
        
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
        
        
        regressor = LinearRegression()
        regressor.fit(x_train,y_train)
        
        regressor.intercept_
        
        regressor.coef_
        
        
        y_pred = regressor.intercept_ + x_test*regressor.coef_
        y_pred - y_test
        
        y_pred = regressor.predict(x_test)
        y_pred - y_test

        plt.scatter(x_train, y_train, color = 'red')
        plt.plot(x_train, regressor.predict(x_train), color = 'blue')
        plt.title('Salary vs Experience (Training set)')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.show()
        
        
        plt.scatter(x_test, y_test, color = 'red')
        plt.plot(x_test, regressor.predict(x_test), color = 'blue')
        plt.title('Salary vs Experience (Training set)')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.show()



    
