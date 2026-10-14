import warnings

warnings.filterwarnings(
    "ignore",
    message="Gimbal lock detected",
    category=UserWarning,
    module="scipy.spatial.transform",
)
