import torch
from torch import nn
from torch.nn import functional as F

class NONLocalBlock1D(nn.Module):
    def __init__(self, args, dim_curr, dim_past, latent_dim):
        super(NONLocalBlock1D, self).__init__()

        self.in_dim1 = dim_curr
        self.in_dim2 = dim_past
        self.latent_dim =  latent_dim

        self.scale = args.scale
        self.scale_factor = args.scale_factor
        self.dropout_rat = args.dropout_rate
        self.video_feat_dim = args.video_feat_dim
#        self.sub_sample = args.sub_sample

        # init_params1 is used in theta, phi, and g.
        self.initnn = True
        self.initnn2 = False

        # theta = model.ConvNd(in_blob1, prefix + '_theta', in_dim1, latent_dim, [1, 1, 1], strides=[1, 1, 1], pads=[0, 0, 0] * 2, **init_params1)
        self.theta = nn.Conv1d(in_channels=self.in_dim1,
                               out_channels=self.latent_dim,
                               kernel_size=1, stride=1, padding=0)
        if self.initnn:
            nn.init.normal_(self.theta.weight, mean=0, std=0.01)
            nn.init.constant_(self.theta.bias, 0)

        # phi = model.ConvNd( in_blob2, prefix + '_phi', in_dim2, latent_dim, [1, 1, 1], strides=[1, 1, 1], pads=[0, 0, 0] * 2, **init_params1)
        self.phi = nn.Conv1d(in_channels=self.in_dim2,
                             out_channels=self.latent_dim,
                             kernel_size=1, stride=1, padding=0)
        if self.initnn  :
            nn.init.normal_(self.phi.weight, mean=0, std=0.01)
            nn.init.constant_(self.phi.bias, 0)

        # g = model.ConvNd( in_blob2, prefix + '_g', in_dim2, latent_dim, [1, 1, 1], strides=[1, 1, 1], pads=[0, 0, 0] * 2, **init_params1)
        self.g = nn.Conv1d(in_channels=self.in_dim2,
                           out_channels=self.latent_dim,
                           kernel_size=1, stride=1, padding=0)
        if self.initnn   :
            nn.init.normal_(self.g.weight, mean=0, std=0.01)
            nn.init.constant_(self.g.bias, 0)

        # if cfg.FBO_NL.SCALE: theta_phi = model.Scale( theta_phi, theta_phi, scale=latent_dim**-.5)
        if self.scale :
            self.scale_factor = torch.tensor([self.latent_dim**self.scale_factor],
                                             requires_grad=True).to('cuda')

        # """Pre-activation style non-linearity."""
        # x = model.LayerNorm( x, [x + "_ln", x + "_ln_mean", x + "_ln_std"])[0]
        # model.Relu(x, x + "_relu")
        # blob_out = model.ConvNd( blob_out, prefix + '_out', latent_dim, in_dim1, [1, 1, 1], strides=[1, 1, 1], pads=[0, 0, 0] * 2,  **init_params2)
        # blob_out = model.Dropout( blob_out, blob_out + '_drop', ratio= 0.2, is_test=False)
        self.final_layers = nn.Sequential(
                            nn.LayerNorm(torch.Size([self.latent_dim, self.video_feat_dim])),
                            nn.ReLU(),
                            nn.Conv1d(in_channels=self.latent_dim,
                                      out_channels=self.in_dim1,
                                      kernel_size=1, stride=1, padding=0),
                            nn.Dropout(p=self.dropout_rat),
        )
        if self.initnn2   :
            nn.init.constant_(self.final_layers[2].weight, 0)
            nn.init.constant_(self.final_layers[2].bias, 0)

#        if self.sub_sample:
#            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
#            self.g = nn.Sequential(self.g, max_pool_layer)
#            self.phi = nn.Sequential(self.phi, max_pool_layer)


    def forward(self, x_past, x_curr):
        # x_curr.shape = torch.Size([64, 5, 400])
        # x_past.shape = torch.Size([64, 25, 400])

#-------------------------------------------------------------------------------1
        # theta = model.ConvNd(in_blob1, prefix + '_theta', in_dim1, latent_dim, [1, 1, 1], strides=[1, 1, 1], pads=[0, 0, 0] * 2, **init_params1)
        # theta, theta_shape_5d = model.Reshape(theta, [theta + '_re', theta + '_shape5d'],
        #                                       shape=(-1, latent_dim, num_feat1))
        theta_x = self.theta(x_curr)        # theta_x.shape = torch.Size([64, 1024, 400])
        theta_x = theta_x.permute(0, 2, 1)  # theta_x.shape = torch.Size([64, 400, 1024])

#-------------------------------------------------------------------------------2
        # phi = model.ConvNd( in_blob2, prefix + '_phi', in_dim2, latent_dim, [1, 1, 1], strides=[1, 1, 1], pads=[0, 0, 0] * 2, **init_params1)
        # phi, phi_shape_5d = model.Reshape(phi, [phi + '_re', phi + '_shape5d'],
        #                                   shape=(-1, latent_dim, num_feat2))
        phi_x = self.phi(x_past)   # phi_x.shape = torch.Size([64, 1024, 400])

#-------------------------------------------------------------------------------3
        # g = model.ConvNd( in_blob2, prefix + '_g', in_dim2, latent_dim, [1, 1, 1], strides=[1, 1, 1], pads=[0, 0, 0] * 2, **init_params1)
        # g, g_shape_5d = model.Reshape(g, [g + '_re', g + '_shape5d'],
        #                               shape=(-1, latent_dim, num_feat2))
        g_x = self.g(x_past)         # g_x.shape = torch.Size([64, 1024, 400])
        g_x = g_x.permute(0, 2, 1)   # g_x.shape = torch.Size([64, 400, 1024])

#-------------------------------------------------------------------------------4
        # (N, C, num_feat1), (N, C, num_feat2) -> (N, num_feat1, num_feat2)
        # theta_phi = model.net.BatchMatMul( [theta, phi], prefix + '_affinity', trans_a=1)
        theta_phi = torch.matmul(theta_x, phi_x)  # theta_phi.shape = torch.Size([64, 400, 400]) = torch.Size([64, 400, 1024]) * torch.Size([64, 1024, 400])

#-------------------------------------------------------------------------------5
        # if cfg.FBO_NL.SCALE: theta_phi = model.Scale( theta_phi, theta_phi, scale=latent_dim**-.5)
        if self.scale :
            theta_phi = theta_phi * self.scale_factor  # theta_phi.shape = torch.Size([64, 400, 400])

#-------------------------------------------------------------------------------6
        # p = model.Softmax(theta_phi, theta_phi + '_prob', engine='CUDNN', axis=2)
        p_x = F.softmax(theta_phi, dim=-1)   # p_x.shape = torch.Size([64, 400, 400])

#-------------------------------------------------------------------------------7
        # (N, C, num_feat2), (N, num_feat1, num_feat2) -> (B, C, num_feat1)
        # t = model.net.BatchMatMul([g, p], prefix + '_y', trans_b=1)
        t_x = torch.matmul(p_x, g_x)  # t_x.shape = torch.Size([64, 400, 1024]) = torch.Size([64, 400, 1024])* torch.Size([64, 400, 400])

#-------------------------------------------------------------------------------8
        # blob_out, t_shape = model.Reshape( [t, theta_shape_5d], [t + '_re', t + '_shape3d'])
        t_x = t_x.permute(0, 2, 1).contiguous() # t_x.shape = torch.Size([64, 1024, 400])

#-------------------------------------------------------------------------------8
        # """Pre-activation style non-linearity."""
        # x = model.LayerNorm( x, [x + "_ln", x + "_ln_mean", x + "_ln_std"])[0]
        # model.Relu(x, x + "_relu")
        # blob_out = model.ConvNd( blob_out, prefix + '_out', latent_dim, in_dim1, [1, 1, 1], strides=[1, 1, 1], pads=[0, 0, 0] * 2,  **init_params2)
        # blob_out = model.Dropout( blob_out, blob_out + '_drop', ratio= 0.2, is_test=False)
        W_t = self.final_layers(t_x) # W_t.shape = torch.Size([64, 5, 400])

        z_x = W_t + x_curr           # z_x.shape = torch.Size([64, 5, 400])
        return z_x



