import unittest
import torch
import pytorch_tcn
from pytorch_tcn import TCN

import inspect
import itertools

def generate_combinations(test_args):
    combinations = []

    for x in test_args:
        kwargs = x['kwargs']
        # kwargs contains a list of values for each key
        # Get all possibe combinations of the values:
        keys = kwargs.keys()
        values = kwargs.values()

        for value_combination in itertools.product(*values):
            combination_dict = dict(zip(keys, value_combination))
            combinations.append(
                dict(
                    kwargs = combination_dict,
                    expected_error = x['expected_error'],
                    )
                )

    return combinations

def get_optional_parameters(cls, method_name):
    sig = inspect.signature(getattr(cls, method_name))
    return [name for name, param in sig.parameters.items() if param.default != inspect.Parameter.empty]

class TestTCN(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        self.available_activations = pytorch_tcn.tcn.activation_fn.keys()
        self.available_norms = TCN(10,[10]).allowed_norm_values
        self.available_initializers = pytorch_tcn.tcn.kernel_init_fn.keys()

        self.num_inputs = 20
        self.num_channels = [
            32, 64, 64, 128,
            32, 64, 64, 128,
            ]
        
        self.batch_size = 10
        self.time_steps = 196
        
        self.test_args = [
            # Test different kernel sizes
            dict(
                kwargs = dict( kernel_size = [3, 5, 7] ),
                expected_error = None,
            ),
            # Test valid dilation rates
            dict(
                kwargs = dict(
                    dilations = [
                        [1, 2, 3, 4, 1, 2, 3, 4],
                        None,
                    ],
                ),
                expected_error = None,
            ),
            # Test invalid dilation rates
            dict(
                kwargs = dict( dilations = [ [1, 2, 3] ] ),
                expected_error = ValueError,
            ),
            # Test valid dilation reset values
            dict(
                kwargs = dict( dilation_reset = [4, None] ),
                expected_error = None,
            ),
            dict(
                kwargs = dict( dropout = [0.0, 0.5], ),
                expected_error = None,
            ),
            dict(
                kwargs = dict( causal = [True, False], ),
                expected_error = None,
            ),
            dict(
                kwargs = dict(
                    use_norm = self.available_norms,
                    use_gate = [True, False],
                    ),
                expected_error = None,
            ),
            dict(
                kwargs = dict( use_norm = [ 'invalid' ] ),
                expected_error = ValueError,
            ),
            dict(
                kwargs = dict(
                    activation = self.available_activations,
                    use_gate = [True, False],
                    ),
                expected_error = None,
            ),
            dict(
                kwargs = dict( activation = [ 'invalid' ] ),
                expected_error = ValueError,
            ),
            dict(
                kwargs = dict( kernel_initializer = self.available_initializers ),
                expected_error = None,
            ),
            dict(
                kwargs = dict( kernel_initializer = [ 'invalid' ] ),
                expected_error = ValueError,
            ),
            dict(
                kwargs = dict( use_skip_connections = [True, False], ),
                expected_error = None,
            ),
            dict(
                kwargs = dict( input_shape = ['NCL', 'NLC'] ),
                expected_error = None,
            ),
            # Test valid embedding shapes
            dict(
                kwargs = dict(
                    embedding_shapes = [
                        [ (10,), ],
                        [ (10,), (128,), ],
                        [ (1, None,), ],
                        [ (32,), (12, None,), ],
                        None,
                    ],
                    embedding_mode = [ 'concat', 'add' ],
                    use_gate = [True, False],
                ),
                expected_error = None,
            ),
            # Test invalid embedding shapes
            dict(
                kwargs = dict(
                    embedding_shapes = [
                        [ (10, 32, 64), ],
                        [ (10, self.time_steps + 32),],
                    ],
                    embedding_mode = [ 'concat', 'add' ],
                    use_gate = [True, False],
                ),
                expected_error = ValueError,
            ),
        ]

        self.combinations = generate_combinations(self.test_args)

        return

    def test_tcn(self, **kwargs):

        tcn = TCN(
            num_inputs = self.num_inputs,
            num_channels = self.num_channels,
            **kwargs,
        )

        x = torch.randn(
            self.batch_size,
            self.num_inputs,
            self.time_steps,
            )
        expected_shape = (
            self.batch_size,
            self.num_channels[-1],
            self.time_steps,
            )

        # check if 'input_shape' is 'NCL'
        if 'input_shape' in kwargs and kwargs['input_shape'] == 'NLC':
            x = x.permute(0, 2, 1)
            expected_shape = (
                self.batch_size,
                self.time_steps,
                self.num_channels[-1],
                )

        if 'embedding_shapes' in kwargs and kwargs['embedding_shapes'] is not None:
            embeddings = []
            for shape in kwargs[ 'embedding_shapes' ]:
                if None in shape:
                    # replace None with self.time_steps
                    shape = list(shape)
                    shape[ shape.index(None) ] = self.time_steps
                    shape = tuple(shape)

                embeddings.append(
                    torch.randn(
                        self.batch_size,
                        *shape,
                        )
                    )
        else:
            embeddings = None

        y = tcn(x, embeddings = embeddings)
        
        self.assertEqual( y.shape, expected_shape )
        return
    
    def test_tcn_grid_search(self):

        # Test all valid combinations
        for test_dict in self.combinations:
            kwargs = test_dict['kwargs']
            print( 'Testing kwargs: ', kwargs )
            if test_dict['expected_error'] is None:
                self.test_tcn( **kwargs )
            else:
                with self.assertRaises(test_dict['expected_error']):
                    self.test_tcn( **kwargs )

        return
    
    def test_if_all_args_get_tested(self):
        # Get kwargs of TCN class
        tcn_optional_parameters = get_optional_parameters(TCN, '__init__')
        print( 'Test if allvariable names of tcn get tested: ', tcn_optional_parameters )
        found_params = { x: False for x in tcn_optional_parameters }

        # check that all tcn_kwargs are there as keys in test_args
        for kwarg in tcn_optional_parameters:
            for x in self.test_args:
                kwargs = x['kwargs']
                if kwarg in kwargs.keys():
                    found_params[kwarg] = True
                    break
        print( 'Params that get tested: ', found_params )
        all_params_found = all( found_params.values() )
        self.assertTrue(all_params_found)
        return
   

if __name__ == '__main__':
    unittest.main()