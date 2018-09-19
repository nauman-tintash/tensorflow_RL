import gym
import numpy as np
import time
import os

import tensorflow as tf

def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1

    # Convert from 80 x 80 matrix to 1600 x 1 matrix
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
    random_value = np.random.uniform()
    if random_value < probability[0]:
        # signifies up in openai gym
        return 2
    else:
         # signifies down in openai gym
        return 3

def discount_with_rewards(episode_rewards, gamma):
    # discounted_rewards = np.zeros_like(episode_rewards)
    # for t in range(len(episode_rewards)):
    #     discounted_reward_sum = 0
    #     discount = 1
    #     for k in range(t, len(episode_rewards)):
    #         discounted_reward_sum += episode_rewards[k] * discount
    #         discount *= gamma
    #         if episode_rewards[k] != 0:
    #             # Don't count rewards from subsequent rounds
    #             break
    #     discounted_rewards[t] = discounted_reward_sum

    discounted_rewards = np.zeros_like(episode_rewards)
    running_add = 0
    for t in reversed(range(0, len(episode_rewards))):
        if episode_rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + episode_rewards[t]
        discounted_rewards[t] = running_add

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    return discounted_rewards


def main():
    num_hidden_layer_neurons = 200
    learning_rate = 0.0005
    checkpoint_every_n_episodes = 100
    resume = True
    gamma = 0.99 # discount factor for reward
    render = True
    
    env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image

    episode_number = 0
    input_dimensions = 80 * 80

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
        units=1,
        activation=tf.sigmoid,
        kernel_initializer=tf.contrib.layers.xavier_initializer())

    # +1 for up, -1 for down
    sampled_actions = tf.placeholder(tf.float32, [None, 1])
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

    checkpoint_file = os.path.join('checkpoints','policy_network.ckpt')
    saver = tf.train.Saver()

    if resume:
        saver.restore(session, checkpoint_file)

    while True:
        if render :
            time.sleep(0.03)
            env.render()
            
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)

        up_probability = session.run(
            output_layer,
            feed_dict={observation_placeholder: processed_observations.reshape([1, -1])}) #Review later-------------------------------------------------------------
    
        action = choose_action(up_probability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)
    
        reward_sum += reward

        # fake_label = 1 if action == 2 else 0
        # loss_function_gradient = fake_label - up_probability
        # episode_gradient_log_ps.append(loss_function_gradient)

        if action == 2:
            action = 1
        else:
            action = 0

        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_observations.append(processed_observations)

        if done: 
            episode_number += 1
            
            discounted_reward = discount_with_rewards(episode_rewards, gamma)
            
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
                print ('ep %d: game finished, reward: %f' % (episode_number, reward_sum))

            reward_sum = 0
            prev_processed_observations = None

            
            if episode_number % checkpoint_every_n_episodes == 0:
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