import gym
import numpy as np
import time
import os

import tensorflow as tf

# import matplotlib.pyplot as plt

def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 192] = 0
    # image[image == 109] = 0
    return image

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    # plt.imshow(input_observation)
    # plt.show()
    """ convert the 210x160x3 uint8 frame into a 3600 float vector """
    processed_observation = input_observation[60:180] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[np.logical_and(processed_observation != 0, processed_observation != 84)] = 1 # everything else (paddles, ball) just set to 1
    processed_observation[processed_observation == 84 ] = 2 # everything else (paddles, ball) just set to 1

    # plt.imshow(processed_observation)
    # plt.show()

    # Convert from 60 x 80 matrix to 4800 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations

def choose_action(probability):
    probability = probability[0]
    prob0 = probability[0]
    prob1 = prob0 + probability[1]
    prob2 = prob1 + probability[2]
    prob3 = prob2 + probability[3]
    prob4 = prob3 + probability[4]
    random_value = np.random.uniform(0.0, prob4)
    
    if random_value < prob0:
        # signifies up in openai gym
        return 2
    elif random_value < prob1:
         # signifies down in openai gym
        return 5
    if random_value < prob2:
        # signifies left in openai gym
        return 3
    elif random_value < prob3:
         # signifies right in openai gym
        return 4
    else:
        # signifies none in openai gym
        return 0

def discount_with_rewards(episode_rewards, gamma):
    discounted_rewards = np.zeros_like(episode_rewards)
    for t in range(len(episode_rewards)):
        discounted_reward_sum = 0
        discount = 1
        for k in range(t, len(episode_rewards)):
            discounted_reward_sum += episode_rewards[k] * discount
            discount *= gamma
            if episode_rewards[k] != 0:
                # Don't count rewards from subsequent rounds
                break
        discounted_rewards[t] = discounted_reward_sum

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    return discounted_rewards


def main():
    num_hidden_layer_neurons = 300
    learning_rate = 0.0005
    batch_size = 1
    checkpoint_every_n_episodes = 10
    resume = True
    gamma = 0.98 # discount factor for reward
    render = True
    
    input_dimensions = 60 * 80

    env = gym.make("Frostbite-v0")
    observation = env.reset() # This gets us the image

    processed_observations, prev_processed_observations = preprocess_observations(observation, None, input_dimensions)

    episode_number = 0

    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    session = tf.InteractiveSession()

    observation_placeholder = tf.placeholder(tf.float32,
                                           [None, input_dimensions])

    hidden_layer = tf.layers.dense(
        observation_placeholder,
        units=num_hidden_layer_neurons,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer())

    output_layer = tf.layers.dense(
        hidden_layer,
        units=5,
        activation=tf.sigmoid,
        kernel_initializer=tf.contrib.layers.xavier_initializer())

    # [up, down, left, right, nothing]
    sampled_actions = tf.placeholder(tf.float32, [None, 5])
    advantage = tf.placeholder(
        tf.float32, [None, 1], name='advantage')

    loss = tf.losses.log_loss(
        labels=sampled_actions,
        predictions=output_layer,
        weights=advantage)
        
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    tf.global_variables_initializer().run(session = session)
        
    episode_observations, episode_rewards, episode_actions = [], [], []

    checkpoint_file = os.path.join('checkpoints_frost','policy_network.ckpt')
    saver = tf.train.Saver()
    # saver.restore(session, checkpoint_file)

    prev_lives = 4
    while True:
        if render :
            time.sleep(0.03)
            env.render()
            
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        # hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)

        up_probability = session.run(
            output_layer,
            feed_dict={observation_placeholder: processed_observations.reshape([1, -1])}) #Review later-------------------------------------------------------------
    
        
        # episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        # print (action)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)
        if(info['ale.lives'] < prev_lives) or done:
            reward = -30
            prev_lives = info['ale.lives']

        # if reward!=0:
        #     print('reward : ', reward)
        reward_sum += reward

        # fake_label = 1 if action == 2 else 0
        # loss_function_gradient = fake_label - up_probability
        # episode_gradient_log_ps.append(loss_function_gradient)

        action_onehot = np.zeros(5)
        if action == 2:
            action_onehot[0] = 1
        elif action == 5:
            action_onehot[1] = 1
        elif action == 3:
            action_onehot[2] = 1
        elif action == 4:
            action_onehot[3] = 1
        else:
            action_onehot[4] = 1
            
        episode_actions.append(action_onehot)
        episode_rewards.append(reward)
        episode_observations.append(processed_observations)

        if done: 

            prev_lives = 4

            episode_number += 1
            
            # if episode_number % batch_size == 0:
            discounted_reward = discount_with_rewards(episode_rewards, gamma)
            # update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

            states_stack = np.vstack(episode_observations)
            actions_stack = np.vstack(episode_actions)
            rewards_stack = np.vstack(discounted_reward)

            feed_dict = {
                observation_placeholder: states_stack,
                sampled_actions: actions_stack,
                advantage: rewards_stack
            }
            session.run(train_op, feed_dict)

            episode_observations, episode_rewards, episode_actions = [], [], [] # reset values
            
            observation = env.reset() # reset env
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

            print(episode_number)
            
            if episode_number % 20 == 0:
                print (' episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
                # print ('ep %d: game finished, reward: %f' % (episode_number, reward_sum))

            reward_sum = 0
            prev_processed_observations = None

            
            if episode_number % 100 == 0:
                saver.save(session, checkpoint_file)
                print('-------------------------SAVED-------------------------------')

            # else:        
            #     episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] # reset values
            #     observation = env.reset() # reset env
            #     running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            #     print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            #     reward_sum = 0
            #     prev_processed_observations = None

    env.close()

        
main()