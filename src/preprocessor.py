from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def Preprocessor (train_X: pd.DataFrame) -> ColumnTransformer :
    """Returns a preprocessor from a the training features dataset"""

    # series description of columns are of dtype object
    L = (train_X.dtypes == 'object')

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
    
    return ColumnTransformer(
        transformers = [
            ('num', numerical_transformer, numerical_cols), # need to define numerical_cols and categorical_cols
            ('cat', categorical_transformer, categorical_cols)
        ])
