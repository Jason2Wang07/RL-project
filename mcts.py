import copy
import math

class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # move -> TreeNode
        self._n_visits = 0   # N
        self._Q = 0          # Q (平均价值)
        self._u = 0          # U (探索加成)
        self._P = prior_p    # P (先验概率)

    def expand(self, action_priors):
        # 展开叶子节点
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        # 选择 UCB 值最大的子节点
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        # 反向传播更新
        # leaf_value 是对于当前节点玩家的价值。
        # 比如：我是黑棋，在这个节点，子节点传回来说黑棋赢了(+1)，那我的Q就要增加。
        self._n_visits += 1
        # Q = (TotalValue) / N
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def get_value(self, c_puct):
        # 计算 PUCT 值: Q + U
        self._u = (c_puct * self._P * math.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return len(self._children) == 0

class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        self._root = None
        self._policy = policy_value_fn # 这是一个函数，输入 board，输出 (probs, value)
        self._c_puct = c_puct
        self._n_playout = n_playout

    def get_move_probs(self, board, temp=1e-3):
        # 进行 n_playout 次模拟
        for _ in range(self._n_playout):
            self._playout(copy.deepcopy(board))

        # 根据根节点的访问次数 N 计算概率
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        
        # 使用温度参数 temp
        act_probs = list(map(lambda x: x**(1.0/temp), visits))
        prob_sum = sum(act_probs)
        probs = [x / prob_sum for x in act_probs]
        
        return acts, probs

    def _playout(self, board):
        node = self._root
        
        # 1. Selection (一直走到叶子节点)
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            board.do_move(action)

        # 检查此时游戏是否结束
        winner = board.check_winner()
        
        # 2. Evaluation (使用神经网络)
        if winner is None:
            # 游戏未结束，询问神经网络
            action_probs, leaf_value = self._policy(board)
            # 3. Expansion (展开新节点)
            # 过滤掉非法动作
            valid_moves = board.get_legal_moves()
            valid_probs = []
            for move in valid_moves:
                valid_probs.append((move, action_probs[move])) 
            node.expand(valid_probs)
        else:
            # 游戏结束
            if winner == 0:
                leaf_value = 0
            else:
                # 如果当前 board.current_player 是赢家，则是1，否则是-1
                # 注意：check_winner 返回的是 1 或 -1 (黑或白)
                # board.current_player 是刚才被做动作切换后的玩家
                leaf_value = 1.0 if winner == board.current_player else -1.0

        # 4. Backpropagation (反向传播)
        # 注意符号反转：每一层父节点看到的价值是子节点的相反数
        node.update(leaf_value)
        # 递归更新父节点
        while node._parent:
            node = node._parent
            leaf_value = -leaf_value
            node.update(leaf_value)

    def update_with_move(self, last_move):
        # 复用子树：如果上一步的搜索树里已经有了这一步的子节点，直接继承
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)
            
    def reset(self):
        self._root = TreeNode(None, 1.0)