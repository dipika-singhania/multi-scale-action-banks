from torch import nn
from  non_local_embedded_gaussian import NONLocalBlock1D
import torch
import torch.nn.functional as F



class Network_sunNON(nn.Module):
    def __init__(self, args, in_dim_curr, in_dim_past):
        super(Network_sunNON, self).__init__()

        self.dropout_rate = args.dropout_rate_linear

        self.video_feat_dim  = args.video_feat_dim
        self.latent_dim      = args.latent_dim
        self.linear_dim      = args.linear_dim

        self.in_dim_curr  =   in_dim_curr
        self.in_dim_past  =   in_dim_past

        # all past
        self.convBlockPast1_all = NONLocalBlock1D(args, self.in_dim_past, self.in_dim_past, self.latent_dim)
#        self.convBlockPast2_all = NONLocalBlock1D(args, self.in_dim_past, self.in_dim_past, self.latent_dim)
#        self.convBlockPast3_all = NONLocalBlock1D(args, self.in_dim_past, self.in_dim_past, self.latent_dim)
        self.convBlock1_all =     NONLocalBlock1D(args, self.in_dim_curr, self.in_dim_past, self.latent_dim)
#        self.convBlock2_all =     NONLocalBlock1D(args, self.in_dim_curr, self.in_dim_past, self.latent_dim)
#        self.convBlock3_all =     NONLocalBlock1D(args, self.in_dim_curr, self.in_dim_past, self.latent_dim)

        self.fc_future_all = nn.Sequential(
            nn.Linear(in_features  = 2*self.in_dim_curr*self.video_feat_dim ,
                      out_features = self.linear_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features  = self.linear_dim,
                      out_features = self.linear_dim)
        )
        self.fc_task_all = nn.Sequential(
            nn.Linear(in_features  = self.in_dim_past*self.video_feat_dim +  2*self.in_dim_curr*self.video_feat_dim,
                      out_features = self.linear_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features  = self.linear_dim,
                      out_features = self.linear_dim)
        )


    def forward(self, x_past_actual, x_curr_actual):
        # x_curr_actual.shape = torch.Size([64, 5, 400])
        # x_past_actual.shape = torch.Size([64, 25, 400])
        batch_size = x_past_actual.size(0)  #  batch_size = 64

        nle_x_past   = F.relu(self.convBlockPast1_all(x_past_actual,  x_past_actual)) #  x_past.shape = torch.Size([64, 15, 400])
#        nle_x_past   = F.relu(self.convBlockPast2_all(x_past_actual,  nle_x_past))        #  x_past.shape = torch.Size([64, 15, 400])
#        nle_x_past   = F.relu(self.convBlockPast3_all(x_past_actual,  nle_x_past))        #  x_past.shape = torch.Size([64, 15, 400])
        nle_x_future = F.relu(    self.convBlock1_all(nle_x_past,     x_curr_actual))        #  output.shape = torch.Size([64, 5, 400])
#        nle_x_future = F.relu(    self.convBlock2_all(nle_x_past,     nle_x_future))        #  output.shape = torch.Size([64, 5, 400])
#        nle_x_future = F.relu(    self.convBlock3_all(nle_x_past,     nle_x_future))        #  output.shape = torch.Size([64, 5, 400])

        all_x_future = torch.cat((nle_x_future, x_curr_actual), 1)                      #  all_x_future.shape = torch.Size([64, 10, 400])
        all_x_task   = torch.cat((nle_x_past  , all_x_future),  1)                      #  all_x_task.shape = torch.Size([32, 25, 400])

        output_future_fc      = self.fc_future_all(all_x_future.view(batch_size, -1))   # output_future_fc = torch.Size([64, 1024])
        output_task_fc        = self.fc_task_all(  all_x_task.view(batch_size,   -1))   # output_future_fc = torch.Size([64, 1024])

        return output_future_fc, output_task_fc


class NetworkinNetwork(nn.Module):
    def __init__(self, args):
        super(NetworkinNetwork, self).__init__()

        self.linear_dim  = args.linear_dim

        self.in_dim_curr  =  args.in_dim_curr
        self.in_dim_past1 =  args.in_dim_past1
        self.in_dim_past2 =  args.in_dim_past2
        self.in_dim_past3 =  args.in_dim_past3
#        self.in_dim_past4 =  args.in_dim_past4

        self.convNONfc_s1 = Network_sunNON(args, self.in_dim_curr, self.in_dim_past1)
        self.convNONfc_s2 = Network_sunNON(args, self.in_dim_curr, self.in_dim_past2)
        self.convNONfc_s3 = Network_sunNON(args, self.in_dim_curr, self.in_dim_past3)
#        self.convNONfc_s4 = Network_sunNON(args, self.in_dim_curr, self.in_dim_past4)

        self.lin_concat_future = nn.Sequential(
            nn.Linear(in_features=3*self.linear_dim, out_features=self.linear_dim)
        )

    def forward(self, x_past_actual_all, x_curr_actual_all, inds_c ):

        netFuture_s1, netPast_s1  = self.convNONfc_s1( x_past_actual_all[inds_c + 0], x_curr_actual_all[inds_c + 0] )
        netFuture_s2, netPast_s2  = self.convNONfc_s2( x_past_actual_all[inds_c + 1], x_curr_actual_all[inds_c + 1] )
        netFuture_s3, netPast_s3  = self.convNONfc_s3( x_past_actual_all[inds_c + 2], x_curr_actual_all[inds_c + 2] )
#        netFuture_s4, netPast_s4  = self.convNONfc_s4( x_past_actual_all[inds_c + 3], x_curr_actual_all[inds_c + 3] )

        comb_netFuture = torch.cat((   netFuture_s1, netFuture_s2, netFuture_s3), 1)
        comb_netFuture = self.lin_concat_future(comb_netFuture)

        comb_netPast   = torch.stack( (netPast_s1,   netPast_s2,   netPast_s3),   0)
        comb_netPast   = torch.max(comb_netPast, 0)[0]

        return comb_netFuture, comb_netPast


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()

        self.n_classes   = args.num_class
        self.linear_dim  = args.linear_dim

        self.NetInNet1 = NetworkinNetwork(args)
        self.NetInNet2 = NetworkinNetwork(args)
        self.NetInNet3 = NetworkinNetwork(args)

        self.cls_future = nn.Sequential(
            nn.Linear(in_features=2*self.linear_dim , out_features=self.n_classes)
        )
        self.cls_future2 = nn.Sequential(
            nn.Linear(in_features=2*self.linear_dim , out_features=self.n_classes)
        )
        self.cls_future3 = nn.Sequential(
            nn.Linear(in_features=2*self.linear_dim , out_features=self.n_classes)
        )

    def forward(self, x_past_actual_all, x_curr_actual_all):

        comb_netFuture1, comb_netPast1 = self.NetInNet1(x_past_actual_all, x_curr_actual_all, 0 )
        comb_netFuture2, comb_netPast2 = self.NetInNet2(x_past_actual_all, x_curr_actual_all, 3 )
        comb_netFuture3, comb_netPast3 = self.NetInNet3(x_past_actual_all, x_curr_actual_all, 6 )


        comb_netFuture_netPast1 = torch.cat((comb_netFuture1, comb_netPast1), 1 )  # output_future_task_fc.shape  = torch.Size([64, 2048])                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! add class scores as well
        output_final_future     = self.cls_future(comb_netFuture_netPast1)                             # output_final_future.shape    = torch.Size([64, 48])

        comb_netFuture_netPast2 = torch.cat((comb_netFuture2, comb_netPast2), 1 )  # output_future_task_fc.shape  = torch.Size([64, 2048])                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! add class scores as well
        output_final_future2    = self.cls_future2(comb_netFuture_netPast2)                            # output_final_future.shape    = torch.Size([64, 48])

        comb_netFuture_netPast3 = torch.cat(( comb_netFuture3, comb_netPast3), 1 )  # output_future_task_fc.shape  = torch.Size([64, 2048])                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! add class scores as well
        output_final_future3    = self.cls_future3(comb_netFuture_netPast3)                            # output_final_future.shape    = torch.Size([64, 48])
  
  
        return output_final_future , output_final_future2, output_final_future3



    def sample_predict(self, x_past, x_curr, Nsamples=100):
        # Just copies type from x, initializes new vector
        predictions_future = x_past[0].data.new(Nsamples, x_past[0].shape[0], self.n_classes)
        predictions_task   = x_past[0].data.new(Nsamples, x_past[0].shape[0], self.n_classes_tasks)
        for i in range(Nsamples):
            output_final, output_final_task  = self.forward(x_past, x_curr )
            predictions_future[i] = output_final
            predictions_task[i] = output_final_task
        return predictions_future, predictions_task



