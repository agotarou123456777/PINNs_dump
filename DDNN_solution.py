import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from lib import lib_FDM as fdm
from lib import lib_NNsim as nsim




def solve_DDNNs():

    '''
    STEP 1
    解析解を求める
    '''
    gamma = 2 # ダンパーの減衰係数
    omega = 20 # 固有角周波数

    t = np.linspace(0,1,500) # タイムステップを生成(0s-1sの間を500分割)
    t = np.reshape(t,[-1,1])
    x = fdm.analytical_solution(gamma, omega, t) # タイムステップを用いて解析解を計算
    x = np.reshape(x,[-1,1])

    # プロット作成
    plt.plot(t, x, label='Analytic Solution Result')
    plt.title('Analytic Solution Result')
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()
    plt.show()

    # 出力されたグラフ上で'q'を押すと次のプロットへ
    print("Please press Q-key")

    '''
    STEP 2
    解析解から学習用のデータポイントを作成
    '''
    # Data points
    datapoint_list = [i for i in range(0,300,10)] # 学習用データとして抜き出す箇所を設定(0-300要素の間を等間隔(10)で抜き出し)
    t_train_data = tf.gather(t, datapoint_list) # タイムステップデータの抜き出し
    x_train_data = tf.gather(x, datapoint_list) # 解析解データの抜き出し

    # プロット作成
    plt.plot(t, x, label='Analytic Solution Result')
    plt.scatter(t_train_data, x_train_data, color='green', label='Extract Data Point for DDNN trainning')
    plt.title('Analytic Solution Result')
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()
    plt.show()
    
    # 出力されたグラフ上で'q'を押すと次のプロットへ
    print("Please press Q-key")

    '''
    STEP 3
    DDNNの構築と学習
    '''
    # Build DDNNs
    n_input = 1
    n_output = 1
    n_neuron = 32
    n_layer = 4
    epochs = 500
    DDNNs = nsim.DataDrivenNNs(n_input, n_output, n_neuron, n_layer, epochs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()
    early_stopping = nsim.EarlyStopping(patience=200,verbose=1)
    DDNNs.build(optimizer, loss_fn, early_stopping)
    
    # DDNNs Trainning
    DDNNs.train(t_train_data, x_train_data)
    print("DDNN trainning complete!")

    # Create test points AND predict result
    t_DDNN_input = np.linspace(0, 1, 20) # DDNNの結果テスト用の入力データ作成
    x_pred = DDNNs._model.predict(t_DDNN_input) # 予測結果の出力
    
    # プロット作成
    plt.plot(t, x, label='Analytic Solution Result')
    plt.scatter(t_DDNN_input, x_pred, color='green', label='DDNN predict result')
    plt.title('Analytic Solution Result')
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    solve_DDNNs()