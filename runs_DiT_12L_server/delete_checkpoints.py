import os
import shutil


def delete_target_dirs(base_path, targets, dry_run=True):
    deleted_paths = []

    for root, dirs, _ in os.walk(base_path, topdown=True):
        for d in dirs:
            if d in targets:
                full_path = os.path.join(root, d)
                deleted_paths.append(full_path)

                if dry_run:
                    print(f"[DRY-RUN] Would delete: {full_path}")
                else:
                    try:
                        shutil.rmtree(full_path)
                        print(f"[DELETED] {full_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to delete {full_path}: {e}")

    if dry_run:
        print("\nDry-run complete.")
        print(f"Total {len(deleted_paths)} directories would be deleted.")
    else:
        print("\nCleanup complete.")
        print(f"Total {len(deleted_paths)} directories deleted.")

    return deleted_paths


if __name__ == "__main__":

    # ==============================
    # 可配置参数区域
    # ==============================
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))  # 当前脚本所在目录
    TARGET_DIR = ROOT_DIR  # 要清理的根目录（例如 runs_DiT_12L_server）
    DELETE_DIRS = ["epoch_loss_plots", "checkpoints"]  # 要删除的目录名称
    DRY_RUN = False  # True = dry-run（只打印将删除的内容）；False = 实际删除
    # ==============================

    print(f"Starting cleanup under: {TARGET_DIR}")
    print(f"Dry-run mode: {DRY_RUN}")
    print(f"Target directories: {DELETE_DIRS}")
    print("=" * 50)
    delete_target_dirs(TARGET_DIR, DELETE_DIRS, DRY_RUN)
