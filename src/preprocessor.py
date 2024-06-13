from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

class Preprocessor:

    def _init_(self, train_X):
        self.train_X = train_X 

    # series descripting of columns are of dtype object
    L = (self.train_X.dtypes == 'object')

    # Obtain a list of all columns containing categorical variables in the training data
    categorical_cols = list(L[L].index)

    # Create a list of all columns containg numerical variables in the training data 
    numerical_cols = [e for e in L.index if not (e in L[L].index)]

    # define the transformer for numerical data
    numerical_transformer = SimpleImputer(strategy='mean')

    # define the transformer for categoric data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # TODO: create preprocessor class that allows for strategy specification 
    def New (self, numerical_cols, categorical_cols):
      return ColumnTransformer(
        transformers = [
            ('num', self.numerical_transformer, numerical_cols), # need to define numerical_cols and categorical_cols
            ('cat', self.categorical_transformer, categorical_cols)
        ]
    )
