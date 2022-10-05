import numpy as np
import argparse
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn_cell
import math
import pickle
import datetime
import random
from tqdm import tqdm

import Dataset

def parse_args():
	########
	# Parses the NJM arguments.
	#######
	parser = argparse.ArgumentParser(description="Run NJM.")

	parser.add_argument('--dataset', type=int, default=1 ,
						help='Choose the dataset. 1 for Epinions and 2 for Gowalla. Default is 1.')

	parser.add_argument('--dimensions', type=int, default=10,
						help='Number of laten dimensions. Default is 10.')

	parser.add_argument('--epochs', type=int, default=20,
						help='Number of epochs. Default is 20.')
	return parser.parse_args()

def get_parameters(args):

	data_path = "data/epinions"
	train_step = 11
	alpha = 0.1
	beta = 0.01
	u_conter = 4630
	s_counter= 26991
	data_name = "epinions"
	if args.dataset == 2:
		data_path = "data/gowalla"
		train_step = 3
		alpha = 1
		beta = 0.1
		u_conter = 21755
		s_counter = 71139
		data_name = "gowalla"
	return data_path, train_step, alpha, beta,u_conter,s_counter,data_name

def seed_tensorflow(seed=1217):
	import os
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	tf.set_random_seed(seed)	# , disable_gpu = TRUE

	# the below seems configs that work for tf2
	os.environ['TF_DETERMINISTIC_OPS'] = '1'
	os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

	tf.config.threading.set_inter_op_parallelism_threads(1)
	tf.config.threading.set_intra_op_parallelism_threads(1)

def print_message(s, print_log_pth):
	print_log = open(print_log_pth, 'a')
	print(s)
	print(s, file=print_log)
	print_log.close()


class NJM:
	def __init__(self, user_id_N=4630, user_attr_M=26991, item_id_N=26991, item_attr_M=4630, embedding_size=10,
				 batch_size=128,beta = 0.01 ,alpha = 0.1,train_T =11,data_name = "epinions",
				 epoch=20):
		self.train_T = train_T
		self.embedding_size = embedding_size
		self.user_node_N = user_id_N + 1
		self.user_attr_M = user_attr_M + 1
		self.item_node_N = item_id_N + 1
		self.item_attr_M = item_attr_M + 1	# 这个特征维度跟用户数量一样，说明特征记录的是被谁互动过
		self.data_name = data_name
		self.batch_size = batch_size
		self.alphaU = beta
		self.alphaS = alpha
		self.epoch = epoch
	def _initialize_weights(self):
		weights = {

			# -------------------------------------------------------------------------------------------------------------------
			'user_latent': tf.Variable(tf.random_normal([self.user_node_N, self.train_T, self.embedding_size]),
									   name='user_latent'),
			# neighbourhood embedding
			'node_proximity': tf.Variable(
				tf.random_normal([self.user_node_N, self.train_T, self.embedding_size]),
				name='node_proximity'),

			'transformation': tf.Variable(tf.ones([self.user_node_N, self.embedding_size]),
										  name='trans'),

			'item_attr_embeddings': tf.Variable(tf.random_normal([self.item_attr_M, self.embedding_size]),
												name='item_attr'),

			'item_rnn_out_rating': tf.Variable(tf.random_normal([self.embedding_size, self.embedding_size]),
											   name='item_rnn_out'),
			# 消费计算中消费影响u和社交影响p的平衡
			'consumption_balance': tf.Variable(tf.random_normal([self.user_node_N], mean=0.5, stddev=0.5),
											   name='consumption_balance'),
			# 好友计算中消费影响u和社交影响p的平衡
			'link_balance': tf.Variable(tf.truncated_normal([self.user_node_N], mean=0.5, stddev=0.5),
										name='link_balance'),

		}
		biases = {
			'item_out_rating': tf.Variable(tf.constant(0.1, shape=[self.item_node_N, self.embedding_size]), name='b1'),
			# 这个没有用到
			'item_out_link': tf.Variable(tf.constant(0.1, shape=[self.item_node_N, self.embedding_size]), name='b2'),
			#
			'user_static': tf.Variable(tf.constant(0.1, shape=[self.user_node_N, self.embedding_size]), name='b3'),
			'item_static': tf.Variable(tf.constant(0.1, shape=[self.item_node_N, self.embedding_size]), name='b4'),
			# 没用到
			'mlp_embeddings': tf.Variable(tf.constant(0.1, shape=[self.embedding_size]), name='b6'),

			'link_mlp_embeddings': tf.Variable(tf.constant(0.1, shape=[self.user_node_N]), name='b7')

		}
		return weights, biases

	def _init_graph(self):
		time_step = self.train_T

		# Input data.
		self.train_user_id = tf.placeholder(tf.int32, shape=[None])  # batch_size * 1

		self.train_item_id = tf.placeholder(tf.int32, shape=[None])
		self.train_item_id_list = tf.placeholder(tf.int32, shape=[None, time_step])

		self.train_rating_indicator = tf.placeholder(tf.float32, shape=[None, time_step])

		# item在那个时刻被哪个用户打分
		self.train_item_attr = tf.placeholder(tf.float32, shape=[None, time_step, self.user_node_N])
		self.train_rating_label = tf.placeholder(tf.float32, shape=[None, time_step])

		# Test_rating
		self.test_user_id = tf.placeholder(tf.int32, shape=[None])


		self.test_friend_record = tf.placeholder(tf.float32, shape=[None, self.user_node_N])
		self.test_item_id = tf.placeholder(tf.int32, shape=[None])
		self.test_item_id_list = tf.placeholder(tf.int32, shape=[None, time_step + 1])

		self.test_item_attr = tf.placeholder(tf.float32, shape=[None, time_step + 1, self.user_node_N])

		self.test_rating_label = tf.placeholder(tf.float32, shape=[None])

		# link prediction
		self.train_predict_link_label = tf.placeholder(tf.float32, shape=[None, time_step, self.user_node_N])

		self.train_predict_weight = tf.placeholder(tf.float32, shape=[None, time_step, self.user_node_N])

		# test link prediction
		self.link_test_user_id = tf.placeholder(tf.int32, shape=[None])

		#   Variables

		network_weights = self._initialize_weights()
		self.weights, self.biases = network_weights

		# get inint state
		self.ini_social_vector = tf.constant(0.0, shape=[self.batch_size, self.embedding_size])
		self.ini_homophily_vector = tf.constant(0.0, shape=[self.batch_size, self.embedding_size])

		self.ini_social_matrix = tf.constant(0.0, shape=[self.user_node_N, self.embedding_size])
		self.ini_homophily_matrix = tf.constant(0.0, shape=[self.user_node_N, self.embedding_size])

		self.one_nodeN = tf.constant(1.0, shape=[self.user_node_N])

		train_item_id_list = tf.reshape(self.train_item_id_list, [-1])
		#   Model

		# 这一部分是item rnn的
		train_item_attr = tf.reshape(self.train_item_attr, [-1, self.item_attr_M])
		# item特征，记录item跟谁互动过，然后查表得到嵌入
		item_attr_embed = tf.matmul(train_item_attr, self.weights['item_attr_embeddings'])
		# 时间长度×维度
		item_rnn_input = tf.reshape(item_attr_embed, [-1, self.train_T, self.embedding_size])
		# basic cell 输入输出维度一致
		item_cell = rnn_cell.LayerNormBasicLSTMCell(self.embedding_size)

		self.item_init_state = item_cell.zero_state(self.batch_size, dtype=tf.float32)

		with tf.variable_scope("item_lstm"):
			item_output, item_final_states = tf.nn.dynamic_rnn(item_cell, item_rnn_input,
															   initial_state=self.item_init_state,
															   dtype=tf.float32)
			item_output = tf.reshape(item_output, [-1, self.embedding_size])
			# item经过rnn后有一个仿射变换
			item_affine_rating = tf.matmul(item_output, self.weights['item_rnn_out_rating']) + tf.nn.embedding_lookup(
				self.biases[
					'item_out_rating'], train_item_id_list)	# train_item_id_list就是一个train_T长度，每个值都是spotid的向量
			# [bsz,t,d]每个时刻都有一个特征
			item_affine_rating = tf.reshape(item_affine_rating, [-1, time_step, self.embedding_size])
		# u
		user_latent_vector = tf.nn.embedding_lookup(self.weights['user_latent'], self.train_user_id)
		# p
		user_latent_promixity = tf.nn.embedding_lookup(self.weights['node_proximity'], self.train_user_id)

		consumption_weigh = tf.nn.embedding_lookup(self.weights['consumption_balance'], self.train_user_id)

		link_weigh = tf.nn.embedding_lookup(self.weights['link_balance'], self.train_user_id)

		zero_op = tf.constant(0.0)
		one_op = tf.constant(1.0)

		for t in range(self.train_T):
			if t == 0:

				# -------------------------
				# add hidden layers here， 这是模型图的左上角部分
				# user_latent_vector每个时刻都用一个placeholder保存
				user_latent_self = user_latent_vector[:, 0, :]
				# u和q_hair拼接
				embed_layer = tf.concat([user_latent_self, self.ini_social_vector], -1)
				# 输出uhat
				user_output = tf.layers.dense(inputs=embed_layer, units=self.embedding_size,
											  activation=tf.nn.relu,
											  kernel_regularizer=tf.contrib.layers.l2_regularizer(
												  0.1), name='rating_mlp')
				# -------------------------
				# rating prediction

				b_user = tf.nn.embedding_lookup(self.biases['user_static'], self.train_user_id)
				b_item = tf.nn.embedding_lookup(self.biases['item_static'], self.train_item_id)
				# 静态bias初始值为0.1，可以更新。
				# bsz*d,tf.multiply按位乘法，这一部分的在论文objective里面
				rating_prediction = tf.multiply(user_output, item_affine_rating[:, t, :]) + tf.multiply(b_user, b_item)
				rating_prediction = tf.reduce_sum(rating_prediction, axis=-1)# 求和的

				rating_prediction = tf.sigmoid(rating_prediction)
				tf.add_to_collection("predict_loss", tf.reduce_sum(
					tf.square(rating_prediction - self.train_rating_label[:, t]) * self.train_rating_indicator[:, t]))	# 后面这项就是真实标签

				# ----------------------------------
				# link prediction
				# user_n * 2d,这个是右上角的E
				user_embedding_matrix = tf.concat([self.weights['node_proximity'][:, t, :], self.ini_homophily_matrix],
												  -1)
				# bsz*2d
				link_embed_layer = tf.concat([user_latent_promixity[:, t, :], self.ini_homophily_vector], -1)

				link_embed_layer = tf.layers.dense(inputs=link_embed_layer, units=self.embedding_size,
												   activation=tf.nn.relu,
												   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
												   name='link_mlp'
												   )

				link_embedding_matrix = tf.layers.dense(inputs=user_embedding_matrix, units=self.embedding_size,
														activation=tf.nn.relu,
														kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
														name='link_mlp', reuse=True
														)
				# 输出是bsz * user_N
				link_prediction = tf.matmul(link_embed_layer, link_embedding_matrix,
											transpose_b=True) + self.biases['link_mlp_embeddings']

				tf.add_to_collection("predict_loss",
									 self.alphaS * tf.reduce_sum(self.train_predict_weight[:, t] *
																 tf.nn.sigmoid_cross_entropy_with_logits(
																	 labels=self.train_predict_link_label[:, t],
																	 logits=link_prediction)))

				# 第一个step后可以知道当前的好友链接情况了
				# bsz*userN
				self.friend_record = self.train_predict_link_label[:, 0, :]
			else:
				# 左上角部分
				# 下面是公式1
				# T和U的元素乘法
				user_friend_latent_matrix = tf.multiply(self.weights['transformation'],
														self.weights['user_latent'][:, t - 1, :])


				# 下面计算trust score f
				## 当前p乘整体p
				node_proximity = tf.matmul(user_latent_promixity[:, t - 1, :],
										   self.weights['node_proximity'][:, t - 1, :], transpose_b=True)
				## 这里是先过sigmoid然后过邻接矩阵，论文里是反过来
				trust_score = tf.sigmoid(node_proximity)
				# self.friend_record 是I
				trust_score = tf.multiply(self.friend_record, trust_score)
				# 这里做的就是算行和然后做分母
				all = tf.reduce_sum(trust_score, keep_dims=True, axis=-1)

				all_p = all + one_op
				# 下面这行的意思是，all里面等于0的值填上1，因为下面做除法
				all = tf.where(tf.equal(all, zero_op), all_p, all)
				trust_score = tf.div(trust_score, all)	# 这样操作，那么trust score/行sum，相当于变成权重了，全零行还是0
				# 这里就是公式里的f*Q，维度是bsz*d

				user_friend_latent_vector = tf.matmul(trust_score, user_friend_latent_matrix)

				# -------------------------
				# add hidden layers here
				# consumption_weigh是bsz长度，每个用户一个，这里好像是用来表示用户偏好受到消费行为影响的比例，剩下的就是受到好友影响的
				user_latent_self = tf.multiply(consumption_weigh, tf.transpose(user_latent_vector[:, t - 1, :]))
				user_latent_self = tf.transpose(user_latent_self)

				user_friend_latent_vector = tf.multiply((1 - consumption_weigh),
														tf.transpose(user_friend_latent_vector))
				user_friend_latent_vector = tf.transpose(user_friend_latent_vector)
				# 拼接放入线性层
				embed_layer = tf.concat([user_latent_self, user_friend_latent_vector], -1)

				user_output = tf.layers.dense(inputs=embed_layer, units=self.embedding_size,
											  activation=tf.nn.relu,
											  kernel_regularizer=tf.contrib.layers.l2_regularizer(
												  0.1), name='rating_mlp', reuse=True)
				# -------------------------
				# rating prediction

				b_user = tf.nn.embedding_lookup(self.biases['user_static'], self.train_user_id)
				b_item = tf.nn.embedding_lookup(self.biases['item_static'], self.train_item_id)
				rating_prediction = tf.multiply(user_output, item_affine_rating[:, t, :]) + tf.multiply(b_user, b_item)
				rating_prediction = tf.reduce_sum(rating_prediction, axis=-1)

				rating_prediction = tf.sigmoid(rating_prediction)
				tf.add_to_collection("predict_loss", tf.reduce_sum(
					tf.square(rating_prediction - self.train_rating_label[:, t]) * self.train_rating_indicator[:, t]))

				# ----------------------------------
				# link prediction


				homo_effect = user_latent_vector[:, t - 1, :]

				node_proximity_by_weight = tf.multiply(link_weigh, tf.transpose(user_latent_promixity[:, t - 1, :]))
				node_proximity_by_weight = tf.transpose(node_proximity_by_weight)

				homo_effect_by_weight = tf.multiply((1 - link_weigh), tf.transpose(homo_effect))
				homo_effect_by_weight = tf.transpose(homo_effect_by_weight)

				# 计算E
				## 这是P
				user_node_matrix = tf.multiply(self.weights['link_balance'],
											   tf.transpose(self.weights['node_proximity'][:, t - 1, :]))
				user_node_matrix = tf.transpose(user_node_matrix)

				## 这是U
				user_latent_matrix = tf.multiply((self.one_nodeN - self.weights['link_balance']),
												 tf.transpose(self.weights['user_latent'][:, t - 1, :]))
				user_latent_matrix = tf.transpose(user_latent_matrix)

				user_embedding_matrix = tf.concat([user_node_matrix, user_latent_matrix], -1)
				link_embed_layer = tf.concat([node_proximity_by_weight, homo_effect_by_weight], -1)

				# 这里使用一层神经网络计算E，这里输入为2d，输出为userN*d
				user_embedding_matrix = tf.layers.dense(inputs=user_embedding_matrix, units=self.embedding_size,
														activation=tf.nn.relu,
														kernel_regularizer=tf.contrib.layers.l2_regularizer(
															0.1), name='link_mlp', reuse=True)
				# u和p过mlp
				link_embed_layer = tf.layers.dense(inputs=link_embed_layer, units=self.embedding_size,
												   activation=tf.nn.relu,
												   kernel_regularizer=tf.contrib.layers.l2_regularizer(
													   0.1), name='link_mlp', reuse=True)
				# h乘E
				link_prediction = tf.matmul(link_embed_layer, user_embedding_matrix, transpose_b=True) + self.biases[
					'link_mlp_embeddings']

				# label的值是1，根据真实权重加权损失
				tf.add_to_collection("predict_loss",
									 self.alphaS * tf.reduce_sum(self.train_predict_weight[:, t] *
																 tf.nn.sigmoid_cross_entropy_with_logits(
																	 labels=self.train_predict_link_label[:, t],
																	 logits=link_prediction)))
				# 有点像公式11
				tf.add_to_collection("predict_loss", self.alphaU * (tf.reduce_sum
																			   (tf.reduce_sum(
																			   tf.square(
																				   user_output - user_latent_vector[:,
																								 t,
																								 :]))) + tf.reduce_sum
																			   (tf.reduce_sum(
																			   tf.square(user_latent_promixity[:, t - 1,
																						 :] - user_latent_promixity[:,
																							  t, :])))))

				self.friend_record = self.friend_record + self.train_predict_link_label[:, t, :]

		tf.add_to_collection("predict_loss",
							 tf.contrib.layers.l2_regularizer(0.1)(self.weights['user_latent']))
		tf.add_to_collection("predict_loss",
							 tf.contrib.layers.l2_regularizer(0.01)(self.weights['transformation']))
		tf.add_to_collection("predict_loss",
							 tf.contrib.layers.l2_regularizer(0.1)(self.weights['item_attr_embeddings']))
		tf.add_to_collection("predict_loss",
							 tf.contrib.layers.l2_regularizer(0.1)(self.weights['item_rnn_out_rating']))
		tf.add_to_collection("predict_loss",
							 tf.contrib.layers.l2_regularizer(0.001)(self.weights['consumption_balance']))
		tf.add_to_collection("predict_loss",
							 tf.contrib.layers.l2_regularizer(0.1)(self.weights['node_proximity']))
		tf.add_to_collection("predict_loss",
							 tf.contrib.layers.l2_regularizer(0.001)(self.weights['link_balance']))



		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

		tf.add_to_collection("predict_loss", tf.reduce_sum(reg_losses))
		self.predict_rating_loss = tf.add_n(tf.get_collection("predict_loss"))

		# test 部分
		test_item_attr = tf.reshape(self.test_item_attr, [-1, self.item_attr_M])
		test_item_attr_embed = tf.matmul(test_item_attr, self.weights['item_attr_embeddings'])

		test_item_rnn_input = tf.reshape(test_item_attr_embed, [-1, self.train_T + 1, self.embedding_size])
		with tf.variable_scope("test_item_lstm"):
			test_item_output, test_item_final_states = tf.nn.dynamic_rnn(item_cell, test_item_rnn_input,
																		 initial_state=self.item_init_state,
																		 dtype=tf.float32)
			test_item_output = test_item_output[:, -1, :]

			test_item_affine_rating = tf.matmul(test_item_output,
												self.weights['item_rnn_out_rating']) + tf.nn.embedding_lookup(
				self.biases['item_out_rating'], self.test_item_id)


		test_user_friend_latent_matrix = tf.multiply(self.weights['user_latent'][:, -1, :],
													 self.weights['transformation'])

		test_user_latent_vector = tf.nn.embedding_lookup(self.weights['user_latent'], self.test_user_id)
		test_user_latent_promixity = tf.nn.embedding_lookup(self.weights['node_proximity'], self.test_user_id)

		test_user_latent_vector = test_user_latent_vector[:, -1, :]

		test_node_proximity = tf.matmul(test_user_latent_promixity[:, -1, :], self.weights['node_proximity'][:, -1, :],
										transpose_b=True)
		test_trust_score = tf.sigmoid(test_node_proximity)

		test_trust_score = tf.multiply(test_trust_score, self.test_friend_record)

		test_all = tf.reduce_sum(test_trust_score, keep_dims=True, axis=-1)
		test_all_p = test_all + one_op
		test_all = tf.where(tf.equal(test_all, zero_op), test_all_p, test_all)
		test_trust_score = tf.div(test_trust_score, test_all)

		test_user_friend_latent_vector = tf.matmul(test_trust_score, test_user_friend_latent_matrix)

		test_consumption_weigh = tf.nn.embedding_lookup(self.weights['consumption_balance'], self.test_user_id)

		test_social_factor = tf.multiply(test_consumption_weigh, tf.transpose(test_user_friend_latent_vector))
		test_social_factor = tf.transpose(test_social_factor)
		test_embed_layer = tf.concat([test_user_latent_vector, test_social_factor], -1)

		test_user_output = tf.layers.dense(inputs=test_embed_layer, units=self.embedding_size,
										   activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(
				0.1), name='rating_mlp', reuse=True)

		test_b_user = tf.nn.embedding_lookup(self.biases['user_static'], self.test_user_id)
		test_b_item = tf.nn.embedding_lookup(self.biases['item_static'], self.test_item_id)

		test_rating_prediction = tf.multiply(test_user_output, test_item_affine_rating) + tf.multiply(
			test_b_user, test_b_item)
		test_rating_prediction = tf.reduce_sum(test_rating_prediction, axis=-1)

		self.test_rating_prediction = tf.sigmoid(test_rating_prediction)

		link_test_link_weigh = tf.nn.embedding_lookup(self.weights['link_balance'], self.link_test_user_id)

		link_test_user_latent_vector = tf.nn.embedding_lookup(self.weights['user_latent'], self.link_test_user_id)
		link_test_user_latent_promixity = tf.nn.embedding_lookup(self.weights['node_proximity'], self.link_test_user_id)

		link_test_node_proximity = link_test_user_latent_promixity[:, -1, :]

		link_test_homo_effect = link_test_user_latent_vector[:, -1, :]

		link_test_node_proximity_by_weight = tf.multiply(link_test_link_weigh, tf.transpose(link_test_node_proximity))
		link_test_node_proximity_by_weight = tf.transpose(link_test_node_proximity_by_weight)

		link_test_homo_effect_by_weight = tf.multiply((1 - link_test_link_weigh), tf.transpose(link_test_homo_effect))
		link_test_homo_effect_by_weight = tf.transpose(link_test_homo_effect_by_weight)

		# link_test_link_prediction = link_test_node_proximity_by_weight + link_test_homo_effect_by_weight

		link_test_embed_layer = tf.concat([link_test_node_proximity_by_weight, link_test_homo_effect_by_weight], -1)
		link_test_embed_layer = tf.layers.dense(inputs=link_test_embed_layer, units=self.embedding_size,
												activation=tf.nn.relu,
												kernel_regularizer=tf.contrib.layers.l2_regularizer(
													0.1), name='link_mlp', reuse=True)

		test_user_node_matrix = tf.multiply(self.weights['link_balance'],
											tf.transpose(self.weights['node_proximity'][:, -1, :]))
		test_user_node_matrix = tf.transpose(test_user_node_matrix)

		test_user_latent_matrix = tf.multiply((self.one_nodeN - self.weights['link_balance']),
											  tf.transpose(self.weights['user_latent'][:, -1, :]))
		test_user_latent_matrix = tf.transpose(test_user_latent_matrix)

		test_user_embedding_matrix = tf.concat([test_user_node_matrix, test_user_latent_matrix], -1)

		test_user_embedding_matrix = tf.layers.dense(inputs=test_user_embedding_matrix,
													 units=self.embedding_size,
													 activation=tf.nn.relu,
													 kernel_regularizer=tf.contrib.layers.l2_regularizer(
														 0.1), name='link_mlp', reuse=True)

		link_test_link_prediction = tf.matmul(link_test_embed_layer, test_user_embedding_matrix, transpose_b=True) + \
									self.biases['link_mlp_embeddings']

		self.link_test_link_prediction = tf.sigmoid(link_test_link_prediction)

		with tf.variable_scope("train"):

			self.predict_rating_optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
																   epsilon=1e-8).minimize(
				self.predict_rating_loss
			)

	def get_batch_data(self, batch_id):

		user_id_list = []
		spot_id_list = []
		spot_id_list_list = []
		spot_attr_list = []
		rating_indicator_list = []
		rating_pre_list = []
		link_list = []
		link_weight_list = []
		total = len(self.train_data)

		for id in range(self.batch_size):
			if (id + batch_id * self.batch_size) > (total - 1):
				fetch_id = (id + batch_id * self.batch_size) % total
			else:
				fetch_id = id + batch_id * self.batch_size
			one = self.train_data[fetch_id]

			spot_attr = np.zeros((self.train_T, self.user_node_N), dtype='float32')

			spot_attr[0] = np.random.normal(size=(self.user_node_N))
			for record in one['spot_attr']:
				if record[2] < self.train_T - 1:
					spot_attr[record[2] + 1][record[0]] = record[1]

			link = np.zeros((self.train_T, self.user_node_N), dtype='int32')
			link_weight = np.zeros((self.train_T, self.user_node_N), dtype='float32')

			for record in one['link_res']:
				link[record[1]][record[0]] = 1

			for record in one['link_predict_weight']:
				# 有点问题，明明应该包括负采样的边，而且已经记录了权重但是全赋值为1
				link_weight[record[1]][record[0]] = 1.0

			user_id_list.append(one['user_id'])
			spot_id_list.append(one['spot_id'])
			spot_id_list_list.append(np.repeat(one['spot_id'], self.train_T))
			rating_indicator_list.append(one['rating_indicator'])
			# consumption_weight_list.append(consumption_weight)
			spot_attr_list.append(spot_attr)
			rating_pre_list.append(one['rating_pre'])
			link_list.append(link)
			link_weight_list.append(link_weight)

		return user_id_list, spot_id_list, spot_id_list_list, rating_indicator_list, spot_attr_list, rating_pre_list, link_list, link_weight_list

	def get_batch_test_data(self, batch_id):

		user_id_list = []
		spot_id_list = []
		spot_id_list_list = []
		consumption_weight_list = []
		spot_attr_list = []
		rating_pre_list = []
		friend_record_list = []
		# 下面错了，self.test_data是字典
		total = len(self.test_data['ids'])

		for id in range(self.batch_size):
			if (id + batch_id * self.batch_size) > (total - 1):
				fetch_id = (id + batch_id * self.batch_size) % total
			else:
				fetch_id = id + batch_id * self.batch_size
			# print(f"test fetch id:{fetch_id}")
			user_id = self.test_data['ids'][fetch_id][0]
			spot_id = self.test_data['ids'][fetch_id][1]
			spot_list = np.repeat(spot_id, self.train_T + 1)
			rating = self.test_data['rating'][fetch_id]

			spot_attr = np.zeros((self.train_T + 1, self.user_node_N), dtype='float32')
			# link_res = np.zeros((self.train_T , self.user_node_N), dtype='int32')
			friend_record = np.zeros((self.user_node_N), dtype='float32')
			spot_attr[0] = np.random.normal(size=(self.user_node_N))

			if spot_id in self.test_data['spot_dict'].keys():
				for record in self.test_data['spot_dict'][spot_id]:
					spot_attr[record[2] + 1][record[0]] = record[1]
			# if user_id not in self.test_data['links'].keys():
			# 	print(user_id)
			# 	raise KeyError("UserID KeyError")
			if user_id in self.test_data['links'].keys():
				for link in self.test_data['links'][user_id]:
					friend_record[link[0]] = 1.0
			user_id_list.append(user_id)
			spot_id_list.append(spot_id)
			spot_id_list_list.append(spot_list)
			spot_attr_list.append(spot_attr)
			friend_record_list.append(friend_record)
			rating_pre_list.append(rating)
		# print(user_id_list)
		return user_id_list, spot_id_list, spot_id_list_list, spot_attr_list, friend_record_list, rating_pre_list

	def train(self):
		print_log_pth = f"njm_log_{seed}.txt"
		print_log = open(print_log_pth, 'w')
		print_log.close()
		with tf.Session() as self.sess:
			init = tf.global_variables_initializer()
			self.sess.run(init)
			self.sess.run(tf.local_variables_initializer())

			item_rnn_init = self.sess.run(self.item_init_state)
			ini_social_vector = self.sess.run(self.ini_social_vector)
			ini_homophily_vector = self.sess.run(self.ini_homophily_vector)

			training_loss = 0.0

			with open("data/train_"+self.data_name+".pkl", 'rb') as f:
				self.train_data = pickle.load(f)

			self.total_batch = int(len(self.train_data) / self.batch_size)+1

			best_rmse = 1.0
			best_f1 = 0.0
			for epoch in range(self.epoch):
				total_batch = self.total_batch
				starttime = datetime.datetime.now()
				for i in tqdm(range(total_batch)):
					# break
					user_id_list, spot_id_list, spot_id_list_list, rating_indicator_list, spot_attr_list, rating_pre_list, link_list, link_weight_list \
						= self.get_batch_data(i)

					rating_feed_dict = {self.train_user_id: user_id_list,
										self.train_item_id: spot_id_list,
										self.train_item_id_list: spot_id_list_list,
										self.train_rating_indicator: rating_indicator_list,
										self.train_item_attr: spot_attr_list,
										self.train_rating_label: rating_pre_list,
										self.train_predict_link_label: link_list,
										self.train_predict_weight: link_weight_list,
										self.item_init_state: item_rnn_init,
										self.ini_social_vector: ini_social_vector,
										self.ini_homophily_vector: ini_homophily_vector
										}
					rating_loss, rating_opt = self.sess.run((self.predict_rating_loss, self.predict_rating_optimizer),
															feed_dict=rating_feed_dict)

					rating_loss = rating_loss / self.batch_size

					training_loss += rating_loss

					# print("batch:" + str(i) + " finished ")
				tmp = "Epoch:" + f'{epoch + 1:04d}' + ", loss=" + str(training_loss / self.total_batch)
				print_message(tmp, print_log_pth)
				training_loss = 0.0

				with open("data/test_rating_"+self.data_name+".pkl", 'rb') as f:
					self.test_data = pickle.load(f)
				self.total_test_batch = int(len(self.test_data['ids']) / self.batch_size) + 1

				mse = 0.0
				self.test_rating_loss_finial = 0.0
				#
				if epoch > -1:
					print("predicting...")
					for i in tqdm(range(self.total_test_batch)):
						user_id_list, spot_id_list, spot_id_list_list, spot_attr_list, friend_record_list, rating_pre_list = self.get_batch_test_data(
							i)

						test_feed_dict = {self.test_user_id: user_id_list,
										  self.test_item_id: spot_id_list,
										  self.test_item_id_list: spot_id_list_list,
										  self.test_item_attr: spot_attr_list,
										  self.test_rating_label: rating_pre_list,
										  self.test_friend_record: friend_record_list,
										  self.item_init_state: item_rnn_init,
										  self.ini_social_vector: ini_social_vector,
										  self.ini_homophily_vector: ini_homophily_vector
										  }

						prediction = self.sess.run((self.test_rating_prediction), feed_dict=test_feed_dict)
						mse += np.sum(np.square(prediction - rating_pre_list)) / self.batch_size
					rmse  = math.sqrt(mse / self.total_test_batch)
					if rmse < best_rmse:
						best_rmse = rmse
					tmp = "RMSE :" + str(rmse)
					print_message(tmp, print_log_pth)

					endtime = datetime.datetime.now()
					tmp = "training time:" + str((endtime - starttime).seconds) + "s"
					print_message(tmp, print_log_pth)

					# ------- test link-------------
					with open("data/test_link_"+self.data_name+".pkl", 'rb') as f:
						self.test_link = pickle.load(f)
					self.total_link = len(self.test_link['last_pre'])

					precision = 0.0
					recall = 0.0
					# 最后一个step的所有社交关系，{user1:[user2,user3,...],...}
					for user_id in self.test_link['last_pre'].keys():
						# print(user_id)
						if len(self.test_link['last_pre'].keys()) >=1:
							link_feed_dcit = {self.link_test_user_id: [user_id]}
							# 这里输入好像只有userid，应该是直接过mlp打分
							predict_link = self.sess.run(
								(self.link_test_link_prediction),
								feed_dict=link_feed_dcit)
							candidate = np.arange(self.user_node_N - 1)
							candidate = candidate + 1	# 从1开始，跟用户编号对应
							# viewed_link 训练集+测试集所有的好友关系列表
							viewed_link = []
							# last_pre 是从test数据集中获取的user_id的好友列表
							for user_viewed in self.test_link['last_pre'][user_id]:
								if user_viewed not in viewed_link:
									viewed_link.append(user_viewed)
							# till_record 最后一个step之前的好友关系列表
							if user_id in self.test_link['till_record'].keys():
								for user_viewed in self.test_link['till_record'][user_id]:
									if user_viewed not in viewed_link:
										viewed_link.append(user_viewed)
							# 所有用户中没有跟当前用户链接的用户列表
							candidate = np.array([x for x in candidate if x not in viewed_link])
							# print(len(candidate))
							# 随机抽100个负样本
							candidate = random.sample(list(candidate), 100)
							candidate_value = {}

							for user in candidate:
								# 模型对负样本的预测得分
								candidate_value[user] = predict_link[0][user]
							for user in self.test_link['last_pre'][user_id]:
								# 模型对gt的打分
								candidate_value[user] = predict_link[0][user]

							candidate_value = sorted(candidate_value.items(), key=lambda item: item[1], reverse=True)
							y_predict = []
							# 选前5个打分最高的
							for i in range(5):
								y_predict.append(candidate_value[i][0])

							tp = 0.0
							fp = 0
							if len(self.test_link['last_pre'][user_id]) < 5:
								total_ture = len(self.test_link['last_pre'][user_id])
							else:
								total_ture = 5.0
							for y_ in y_predict:
								if y_ in self.test_link['last_pre'][user_id]:
									tp += 1.0
								else:
									fp += 1
							precision += tp / 5.0
							recall += tp /total_ture
					precision = precision / len(self.test_link['last_pre'])
					recall= recall / len(self.test_link['last_pre'])
					f1 = 2*precision*recall / (precision+recall)
					print_message("test precision :" + str(precision), print_log_pth)
					print_message("test recall :" + str(recall), print_log_pth)
					print_message("test f1 :" + str(f1), print_log_pth)

					if f1 > best_f1:
						best_f1  = f1

			print_message("best rmse :" + str(best_rmse), print_log_pth)
			print_message("best f1 :" + str(best_f1), print_log_pth)


if __name__ == "__main__":
	args = parse_args()
	data_path, train_step, alpha, beta,u_counter ,s_counter,data_name = get_parameters(args)
	seed = 5423
	seed_tensorflow(seed)
	data = Dataset.Dataset(train_step=train_step,u_counter=u_counter,s_counter=s_counter,data_name=data_name)
	data.generate()
	njm = NJM(user_id_N=u_counter,user_attr_M=s_counter,item_id_N=s_counter,item_attr_M=u_counter,data_name=data_name,
			  embedding_size=args.dimensions,train_T=train_step,beta=beta, alpha=alpha,epoch=args.epochs)
	njm._init_graph()
	# writer = tf.summary.FileWriter('/path/to/logs', tf.get_default_graph())
	# writer.close()
	# njm.train()


