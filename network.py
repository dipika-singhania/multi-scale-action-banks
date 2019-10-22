from torch import nn
from  non_local_embedded_gaussian import NONLocalBlock1D
import torch
import torch.nn.functional as F



class Network_sunNON(nn.Module):
    def __init__(self, args, dim_curr, dim_past):
        super(Network_sunNON, self).__init__()

        self.dropout_rate = args.dropout_linear

        self.video_feat_dim  = args.video_feat_dim
        self.latent_dim      = args.latent_dim
        self.linear_dim      = args.linear_dim

        self.dim_curr  =   dim_curr
        self.dim_past  =   dim_past

        # all past
#        self.convBlockPast1_all = NONLocalBlock1D(args, self.dim_past, self.dim_past, self.latent_dim)
#        self.convBlockPast2_all = NONLocalBlock1D(args, self.dim_past, self.dim_past, self.latent_dim)
#        self.convBlockPast3_all = NONLocalBlock1D(args, self.dim_past, self.dim_past, self.latent_dim)
        self.convBlock1_all = NONLocalBlock1D(args, self.dim_curr, self.dim_past, self.latent_dim)
#        self.convBlock2_all =     NONLocalBlock1D(args, self.dim_curr, self.dim_past, self.latent_dim)
#        self.convBlock3_all =     NONLocalBlock1D(args, self.dim_curr, self.dim_past, self.latent_dim)

        self.fc_future_all = nn.Sequential(
            nn.Linear(in_features  = 2 * self.dim_curr * self.video_feat_dim ,
                      out_features = self.linear_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features  = self.linear_dim,
                      out_features = self.linear_dim)
        )
        self.fc_task_all = nn.Sequential(
            nn.Linear(in_features  = self.dim_past * self.video_feat_dim +  2 * self.dim_curr * self.video_feat_dim,
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

#        nle_x_past   = F.relu(self.convBlockPast1_all(x_past_actual,  x_past_actual)) #  x_past.shape = torch.Size([64, 15, 400])
#        nle_x_past   = F.relu(self.convBlockPast2_all(x_past_actual,  nle_x_past))        #  x_past.shape = torch.Size([64, 15, 400])
#        nle_x_past   = F.relu(self.convBlockPast3_all(x_past_actual,  nle_x_past))        #  x_past.shape = torch.Size([64, 15, 400])
        nle_x_future = F.relu(    self.convBlock1_all(x_past_actual,     x_curr_actual))        #  output.shape = torch.Size([64, 5, 400])
#        nle_x_future = F.relu(    self.convBlock2_all(nle_x_past,     nle_x_future))        #  output.shape = torch.Size([64, 5, 400])
#        nle_x_future = F.relu(    self.convBlock3_all(nle_x_past,     nle_x_future))        #  output.shape = torch.Size([64, 5, 400])

        all_x_future = torch.cat((nle_x_future, x_curr_actual), 1)                      #  all_x_future.shape = torch.Size([64, 10, 400])
        all_x_task   = torch.cat((x_past_actual  , all_x_future),  1)                      #  all_x_task.shape = torch.Size([32, 25, 400])

        output_future_fc      = self.fc_future_all(all_x_future.view(batch_size, -1))   # output_future_fc = torch.Size([64, 1024])
        output_task_fc        = self.fc_task_all(  all_x_task.view(batch_size,   -1))   # output_future_fc = torch.Size([64, 1024])

        return output_future_fc, output_task_fc


class NetworkinNetwork(nn.Module):
    def __init__(self, args):
        super(NetworkinNetwork, self).__init__()

        self.linear_dim  = args.linear_dim

        self.dim_curr  =  args.dim_curr
        self.dim_past_list = args.dim_past_list
        

        self.convNONfc_list = nn.ModuleList([Network_sunNON(args, self.dim_curr, dim_past) for dim_past in self.dim_past_list])
        # self.dim_past1 =  args.dim_past1
        # self.dim_past2 =  args.dim_past2
        # self.dim_past3 =  args.dim_past3

        # self.convNONfc_s1 = Network_sunNON(args, self.dim_curr, self.dim_past1)
        # self.convNONfc_s2 = Network_sunNON(args, self.dim_curr, self.dim_past2)
        # self.convNONfc_s3 = Network_sunNON(args, self.dim_curr, self.dim_past3)

        self.lin_concat_future = nn.Sequential(
            nn.Linear(in_features = len(self.convNONfc_list) * self.linear_dim, out_features=self.linear_dim)
        )

    def forward(self, x_past_actual_all, x_curr_actual_all, inds_c):

        netFuture_list    = []
        netPast_list      = []
        for i, convNonfc in enumerate(self.convNONfc_list):
            netFuture_s1, netPast_s1  = convNonfc(x_past_actual_all[i], x_curr_actual_all[inds_c])
            netFuture_list.append(netFuture_s1)
            netPast_list.append(netPast_s1)

        # netFuture_s1, netPast_s1  = self.convNONfc_s1( x_past_actual_all[ 0], x_curr_actual_all[inds_c] )
        # netFuture_s2, netPast_s2  = self.convNONfc_s2( x_past_actual_all[ 1], x_curr_actual_all[inds_c] )
        # netFuture_s3, netPast_s3  = self.convNONfc_s3( x_past_actual_all[ 2], x_curr_actual_all[inds_c] )

        # comb_netFuture = torch.cat((   netFuture_s1, netFuture_s2, netFuture_s3), 1)
        comb_netFuture = torch.cat(netFuture_list, 1)
        comb_netFuture = self.lin_concat_future(comb_netFuture)

        # comb_netPast   = torch.stack( (netPast_s1,   netPast_s2,   netPast_s3),   0)
        comb_netPast   = torch.stack(netPast_list,   0)
        comb_netPast   = torch.max(comb_netPast, 0)[0]

        return comb_netFuture, comb_netPast


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()

        self.n_classes   = args.num_class
        self.linear_dim  = args.linear_dim

        self.NetInNet_list   = nn.ModuleList([NetworkinNetwork(args) for _ in range(len(args.curr_sec_list))])
        self.cls_future_list = nn.ModuleList([nn.Sequential( nn.Linear(in_features=2*self.linear_dim , out_features=self.n_classes) ) \
                                              for _ in range(len(args.curr_sec_list))])

        # self.NetInNet1 = NetworkinNetwork(args)
        # self.NetInNet2 = NetworkinNetwork(args)
        # self.NetInNet3 = NetworkinNetwork(args)
        # self.NetInNet4 = NetworkinNetwork(args)

        # self.cls_future = nn.Sequential(
        #     nn.Linear(in_features=2*self.linear_dim , out_features=self.n_classes)
        # )
        # self.cls_future2 = nn.Sequential(
        #     nn.Linear(in_features=2*self.linear_dim , out_features=self.n_classes)
        # )
        # self.cls_future3 = nn.Sequential(
        #     nn.Linear(in_features=2*self.linear_dim , out_features=self.n_classes)
        # )
        # self.cls_future4 = nn.Sequential(
        #     nn.Linear(in_features=2*self.linear_dim , out_features=self.n_classes)
        # )

        self.add_verb_loss = args.add_verb_loss
        self.add_noun_loss = args.add_noun_loss

        if args.add_verb_loss:
            self.cls_future_verb_list = nn.ModuleList([nn.Sequential( nn.Linear(in_features= 2 * self.linear_dim , out_features=args.verb_class) ) \
                                                       for _ in range(len(args.curr_sec_list))])
            # self.cls_future_verb = nn.Sequential(
            #     nn.Linear(in_features=2*self.linear_dim , out_features=args.verb_class)
            # )
            # self.cls_future2_verb = nn.Sequential(
            #     nn.Linear(in_features=2*self.linear_dim , out_features=args.verb_class)
            # )
            # self.cls_future3_verb = nn.Sequential(
            #     nn.Linear(in_features=2*self.linear_dim , out_features=args.verb_class)
            # )
            # self.cls_future4_verb = nn.Sequential(
            #     nn.Linear(in_features=2*self.linear_dim , out_features=args.verb_class)
            # )

        if args.add_noun_loss:
            self.cls_future_noun_list = nn.ModuleList([nn.Sequential(nn.Linear(in_features = 2 * self.linear_dim , out_features=args.noun_class) ) \
                                                       for _ in range(len(args.curr_sec_list))])
            # self.cls_future_noun = nn.Sequential(
            #     nn.Linear(in_features=2*self.linear_dim , out_features=args.noun_class)
            # )
            # self.cls_future2_noun = nn.Sequential(
            #     nn.Linear(in_features=2*self.linear_dim , out_features=args.noun_class)
            # )
            # self.cls_future3_noun = nn.Sequential(
            #     nn.Linear(in_features=2*self.linear_dim , out_features=args.noun_class)
            # )
            # self.cls_future4_noun = nn.Sequential(
            #     nn.Linear(in_features=2*self.linear_dim , out_features=args.noun_class)
            # )

    def forward(self, x_past_actual_all, x_curr_actual_all):

        output_final_future_list = []
        output_verb_future_list  = []
        output_noun_future_list  = []
        for i, NetInNet in enumerate(self.NetInNet_list):
            comb_netFuture, comb_netPast = NetInNet(x_past_actual_all, x_curr_actual_all, i)
            comb_netFuture_netPast       = torch.cat((comb_netFuture, comb_netPast), 1)  # output_future_task_fc.shape  = torch.Size([64, 2048])
            output_final_future          = self.cls_future_list[i](comb_netFuture_netPast)  # output_final_future.shape    = torch.Size([64, 48])
            output_final_future_list.append(output_final_future)
            if self.add_verb_loss:
                output_verb_future       = self.cls_future_verb_list[i](comb_netFuture_netPast)  # output_final_future.shape    = torch.Size([64, 48])
                output_verb_future_list.append(output_final_future)
            if self.add_noun_loss:
                output_noun_future       = self.cls_future_noun_list[i](comb_netFuture_netPast)  # output_final_future.shape    = torch.Size([64, 48])
                output_noun_future_list.append(output_final_future)
            
        #comb_netFuture1, comb_netPast1 = self.NetInNet1(x_past_actual_all, x_curr_actual_all, 0 )
        #comb_netFuture2, comb_netPast2 = self.NetInNet2(x_past_actual_all, x_curr_actual_all, 1 )
        #comb_netFuture3, comb_netPast3 = self.NetInNet3(x_past_actual_all, x_curr_actual_all, 2 )
        #comb_netFuture4, comb_netPast4 = self.NetInNet4(x_past_actual_all, x_curr_actual_all, 3 )

        #comb_netFuture_netPast1 = torch.cat((comb_netFuture1, comb_netPast1), 1 )  # output_future_task_fc.shape  = torch.Size([64, 2048])                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! add class scores as well
        #output_final_future     = self.cls_future(comb_netFuture_netPast1)                             # output_final_future.shape    = torch.Size([64, 48])

        #comb_netFuture_netPast2 = torch.cat((comb_netFuture2, comb_netPast2), 1 )  # output_future_task_fc.shape  = torch.Size([64, 2048])                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! add class scores as well
        #output_final_future2    = self.cls_future2(comb_netFuture_netPast2)                            # output_final_future.shape    = torch.Size([64, 48])

        #comb_netFuture_netPast3 = torch.cat(( comb_netFuture3, comb_netPast3), 1 )  # output_future_task_fc.shape  = torch.Size([64, 2048])                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! add class scores as well
        #output_final_future3    = self.cls_future3(comb_netFuture_netPast3)                            # output_final_future.shape    = torch.Size([64, 48])

        #comb_netFuture_netPast4 = torch.cat(( comb_netFuture4, comb_netPast4), 1 )  # output_future_task_fc.shape  = torch.Size([64, 2048])                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! add class scores as well
        #output_final_future4    = self.cls_future4(comb_netFuture_netPast4)                            # output_final_future.shape    = torch.Size([64, 48])

        #if self.add_verb_loss:
        #    output_verb_future  = self.cls_future_verb(comb_netFuture_netPast1) 
        #    output_verb_future2 = self.cls_future2_verb(comb_netFuture_netPast2)
        #    output_verb_future3 = self.cls_future3_verb(comb_netFuture_netPast3)
        #    output_verb_future4 = self.cls_future4_verb(comb_netFuture_netPast4)
        #else:
        #    output_verb_future  = None
        #    output_verb_future2 = None
        #    output_verb_future3 = None
        #    output_verb_future4 = None

        #if self.add_noun_loss:
        #    output_noun_future  = self.cls_future_noun(comb_netFuture_netPast1) 
        #    output_noun_future2 = self.cls_future2_noun(comb_netFuture_netPast2)
        #    output_noun_future3 = self.cls_future3_noun(comb_netFuture_netPast3)
        #    output_noun_future4 = self.cls_future4_noun(comb_netFuture_netPast4)
        #else:
        #    output_noun_future  = None
        #    output_noun_future2 = None
        #    output_noun_future3 = None
        #    output_noun_future4 = None


        return output_final_future_list , output_verb_future_list, output_noun_future_list


    def sample_predict(self, x_past, x_curr, Nsamples=100):
        # Just copies type from x, initializes new vector
        predictions_future = x_past[0].data.new(Nsamples, x_past[0].shape[0], self.n_classes)
        predictions_task   = x_past[0].data.new(Nsamples, x_past[0].shape[0], self.n_classes_tasks)
        for i in range(Nsamples):
            output_final, output_final_task  = self.forward(x_past, x_curr )
            predictions_future[i] = output_final
            predictions_task[i] = output_final_task
        return predictions_future, predictions_task



