import os

HyperElasticity = {}

# define the geometry
HyperElasticity["plate_length"] = 4.
HyperElasticity["plate_width"] = 1

# define network parameters
HyperElasticity["num_pts_x"] = 200
HyperElasticity["num_pts_y"] = 50
HyperElasticity["num_epoch"] = 1000
HyperElasticity["num_epoch_LBFGS"] = 0
HyperElasticity["print_epoch"] = 100
HyperElasticity["data_type"] = 'float64'
HyperElasticity["normalized"] = True

# define model parameters
HyperElasticity["beam"] = dict()
HyperElasticity["beam"]["E"] = 1e3
HyperElasticity["beam"]["nu"] = 0.25
HyperElasticity["beam"]["state"] = "plane stress"
HyperElasticity["beam"]["param_c1"] = 630
HyperElasticity["beam"]["param_c2"] = -1.2
HyperElasticity["beam"]["param_c"] = 100
HyperElasticity["beam"]["pressure"] = -10.0
HyperElasticity["beam"]["traction_type"] = 'GRF'
# HyperElasticity["beam"]["traction_type"] = 'constant'
HyperElasticity["beam"]["traction_mean"] = -10.0
HyperElasticity["beam"]["traction_var"] = 0.1
HyperElasticity["beam"]["traction_scale"] = 0.08
HyperElasticity["beam"]["numPtsU"] = HyperElasticity["num_pts_x"]
HyperElasticity["beam"]["numPtsV"] = HyperElasticity["num_pts_y"]
HyperElasticity["beam"]['length'] = HyperElasticity["plate_length"]
HyperElasticity["beam"]['width'] = HyperElasticity["plate_width"]

HyperElasticity["fno"] = dict()
HyperElasticity["fno"]["mode1"] = 8
HyperElasticity["fno"]["mode2"] = 8
HyperElasticity["fno"]["width"] = 32
HyperElasticity["fno"]["depth"] = 4
HyperElasticity["fno"]["channels_last_proj"] = 128
HyperElasticity["fno"]["padding"] = 0
HyperElasticity["fno"]["n_train"] = 500
HyperElasticity["fno"]["n_test"] = 50
HyperElasticity["fno"]["n_data"] = HyperElasticity["fno"]["n_train"] + HyperElasticity["fno"]["n_test"]
HyperElasticity["fno"]["batch_size"] = 50
HyperElasticity["fno"]["learning_rate"] = 0.001
HyperElasticity["fno"]["weight_decay"] = 1e-4

# define FEM parameters
HyperElasticity["FEM_data"] = dict()
HyperElasticity["FEM_data"]["num_pts_x"] = HyperElasticity["num_pts_x"]
HyperElasticity["FEM_data"]["num_pts_y"] = HyperElasticity["num_pts_y"]
HyperElasticity["FEM_data"]["energy_type"] = 'mooneyrivlin'
# HyperElasticity["FEM_data"]["energy_type"] = 'neohookean'
HyperElasticity["FEM_data"]["dimension"] = 2

HyperElasticity["dir"] = './data/'
HyperElasticity["path"] = (HyperElasticity["dir"] + 'Hyperelasticity_n' + str(HyperElasticity["fno"]['n_data']) +
                           '_' + HyperElasticity["FEM_data"]["energy_type"] +
                           '_' + str(HyperElasticity['num_pts_x']) + 'X' + str(HyperElasticity['num_pts_y']) + '.npz')
