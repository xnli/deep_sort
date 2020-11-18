# vim: expandtab:ts=4:sw=4


class TrackState:
    # 单个目标跟踪轨迹状态的枚举类型，包含不确定态、确认态、删除态
    # 不确定态：这种状态会在初始化一个Track的时候分配，并且只有连续匹配上n_init帧才会转变为确定态。如果处于不确定态的情况下没有匹配上任何detection, 那就会转变为删除态。todo 确认是不是连续匹配上n_init
    # 确定态： 代表该Track确定处于匹配状态。如果当前Track属于确定态，但是匹配失败连续达到max_age 次数的时候，就会转变为删除态。
    # 删除态： 代表该Track已失效。
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        # 初始状态分布的均值向量和协防差矩阵 todo 位置和速度信息
        self.mean = mean
        self.covariance = covariance
        # 跟踪ID
        self.track_id = track_id
        # hits 代表匹配上的次数, 匹配次数超过n_init ,状态就会从Tentative 变为Confirmed todo 确认hits 是代表连续匹配上的次数还是匹配上的次数。确认的方式是在没有匹配上的时候会不会清零
        # hits 在每次update时进行更新，但只有match的时候才会update
        self.hits = 1
        # 没有用到
        self.age = 1
        # 每次调用predict 函数time_since_update就会+1, 每次调用update 函数就会设置为0
        # time_since_update 表示距离上次update之后没有update的帧数，也即是没有match成功的帧数（只有match的时候才会update）
        self.time_since_update = 0

        # 初始状态
        self.state = TrackState.Tentative
        # 每个track 对应多个feature, 每次更新将最新的feature 添加到列表
        # 这里feature 代表该轨迹在不同帧对应位置通过ReID提取到的特征。注意：如果存放的feature个数过多，会拖慢计算速度。
        # 这里之所以保存列表，而不是更新当前最新的特征，是为了解决目标被遮挡后再次出现的问题，需要从以往帧对应的特征进行匹配。
        self.features = []
        if feature is not None:
            self.features.append(feature)

        # 设置状态切换为Confirmed的阈值n_init，即hits > _n_init时，state置为Confirmed
        self._n_init = n_init
        # 设置最大存活时间, 即当time_since_age > _max_ages时，state置为Deleted
        self._max_age = max_age

    # 将[x,y,a,h] 转换为[t,l,w,h]
    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """
        完成功能
        ----------
        - 更新均值和方差
        - time_since_update 加1

        参数
        ----------
        kf : kalman_filter.KalmanFilter
            从 Trakcer 那里传入卡尔曼滤波实例

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
