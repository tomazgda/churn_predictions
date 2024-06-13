from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# define the transformer for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# define the transformer for categoric data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# TODO: create preprocessor class that allows for strategy specification 
def preprocessor (numerical_cols, categorical_cols):
  return ColumnTransformer(
    transformers = [
        ('num', numerical_transformer, numerical_cols), # need to define numerical_cols and categorical_cols
        ('cat', categorical_transformer, categorical_cols)
    ]
)
