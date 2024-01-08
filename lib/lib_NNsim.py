import numpy as np
import tensorflow as tf

def MLP(n_input, n_output, n_neuron, n_layer, act_fn='tanh'):
    tf.random.set_seed(1234)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=n_neuron,
            activation=act_fn,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            input_shape=(n_input,),
            name='H1')
    ])
    for i in range(n_layer-1):
        model.add(
            tf.keras.layers.Dense(
                units=n_neuron,
                activation=act_fn,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                name='H{}'.format(str(i+2))
            ))
    model.add(
        tf.keras.layers.Dense(
            units=n_output,
            name='output'
        ))
    return model


class EarlyStopping:
    '''
    早期に学習停止するEarly Stoppingクラスの定義
    '''

    def __init__(self, patience=10, verbose=0):
        '''
        Parameters:
            patience(int): 監視するエポック数(デフォルトは10)
            verbose(int): 早期終了の出力フラグ
                          出力(1),出力しない(0)
        '''

        self.epoch = 0 # 監視中のエポック数のカウンターを初期
        self.pre_loss = float('inf') # 比較対象の損失を無限大'inf'で初期化
        self.patience = patience # 監視対象のエポック数をパラメーターで初期化
        self.verbose = verbose # 早期終了メッセージの出力フラグをパラメーターで初期化

    def __call__(self, current_loss):
        '''
        Parameters:
            current_loss(float): 1エポック終了後の検証データの損失
        Return:
            True:監視回数の上限までに前エポックの損失を超えた場合
            False:監視回数の上限までに前エポックの損失を超えない場合
        '''

        if self.pre_loss < current_loss: # 前エポックの損失より大きくなった場合
            self.epoch += 1 # カウンターを1増やす

            if self.epoch > self.patience: # 監視回数の上限に達した場合
                if self.verbose:  # 早期終了のフラグが1の場合
                    print('early stopping')
                return True # 学習を終了するTrueを返す

        else: # 前エポックの損失以下の場合
            self.epoch = 0 # カウンターを0に戻す
            self.pre_loss = current_loss # 損失の値を更新す

        return False



class DataDrivenNNs():
    '''
    DataDriven型のNNクラス
    '''

    def __init__(self, n_input, n_output, n_neuron, n_layer, epochs, act_fn='tanh'):
        '''
        Input : 
        n_input   || インプット数
        n_output  || アウトプット数
        n_neuron  || 隠れ層のユニット数
        n_layer   || 隠れ層の層数
        act_fn    || 活性化関数
        epochs    || エポック数
        '''
        self.n_input = n_input
        self.n_output = n_output
        self.n_neuron = n_neuron
        self.n_layer = n_layer
        self.epochs = epochs
        self.act_fn = act_fn

    def build(self, optimizer, loss_fn, early_stopping):
        self._model = MLP(self.n_input, self.n_output, self.n_neuron, self.n_layer, self.act_fn)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._early_stopping = early_stopping
        return self

    def train_step(self, t_data, x_data):
        with tf.GradientTape() as tape:
            x_pred = self._model(t_data)
            loss = self._loss_fn(x_pred,x_data)
            #print("loss : ", loss)
        self._gradients = tape.gradient(loss,self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(self._gradients, self._model.trainable_variables))
        self._loss_values.append(loss)
        return self

    def train(self, t_data, x_data):
        '''
        学習ループ用の関数
        '''
        self._loss_values = []
        
        # epochに設定しただけの学習ループを実行
        for i in range(self.epochs):
            self.train_step(t_data, x_data)
            if self._early_stopping(self._loss_values[-1]): #early stoppingの場合ループを抜ける
                break


class PhysicsInformedNNs():
    '''
    PINNs(Physics-informed neural networks)型のNNクラス
    '''

    def __init__(self, n_input, n_output, n_neuron, n_layer, epochs, act_fn='tanh'):
        '''
        n_input   : インプット数
        n_output   : アウトプット数
        n_neuron   : 隠れ層のユニット数
        n_layer   : 隠れ層の層数
        act_fn   : 活性化関数
        epochs   : エポック数
        '''
        self.n_input = n_input
        self.n_output = n_output
        self.n_neuron = n_neuron
        self.n_layer = n_layer
        self.epochs = epochs
        self.act_fn = act_fn

    def build(self, optimizer, loss_fn, early_stopping):
        self._model = MLP(self.n_input, self.n_output, self.n_neuron, self.n_layer, self.act_fn)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._early_stopping = early_stopping
        return self

    def train_step(self, t_data, x_data, t_pinn, c, k):
        with tf.GradientTape() as tape_total:
            tape_total.watch(self._model.trainable_variables)
            x_pred = self._model(t_data)
            loss1 = self._loss_fn(x_pred, x_data)
            loss1 = tf.cast(loss1, dtype=tf.float32)

            with tf.GradientTape() as tape2:
                tape2.watch(t_pinn)
                with tf.GradientTape() as tape1:
                    tape1.watch(t_pinn)
                    x_pred_pinn = self._model(t_pinn)
                dx_dt = tape1.gradient(x_pred_pinn, t_pinn)
            dx_dt2 = tape2.gradient(dx_dt, t_pinn)

            dx_dt  = tf.cast(dx_dt, dtype=tf.float32)
            dx_dt2 = tf.cast(dx_dt2, dtype=tf.float32)
            x_pred_pinn = tf.cast(x_pred_pinn, dtype=tf.float32)

            loss_physics = dx_dt2 + c * dx_dt + k * x_pred_pinn
            loss2 = 5.0e-4 * self._loss_fn(loss_physics, tf.zeros_like(loss_physics))
            loss2 = tf.cast(loss2, dtype=tf.float32)

            loss = loss1 + loss2

        self._optimizer.minimize(loss, self._model.trainable_variables, tape=tape_total)
        self._loss_values.append(loss)
        return self

    def train(self, t_data, x_data, t_pinn, c, k):
        self._loss_values = []
        for i in range(self.epochs):
            self.train_step(t_data, x_data, t_pinn, c, k)
            if self._early_stopping(self._loss_values[-1]):
                break
