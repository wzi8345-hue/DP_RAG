"""MinerU API 客户端: 上传 PDF 并获取解析结果。

从原始 mineru_api.py 提取核心逻辑, 去除模块级副作用代码,
封装为可复用的客户端类。
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import zipfile
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def _safe_resp_text(resp: "requests.Response", limit: int = 300) -> str:
    """按 UTF-8 解码响应体, 失败时回退到 latin-1, 避免中文报错乱码。"""
    try:
        text = resp.content.decode("utf-8", errors="replace")
    except Exception:
        text = resp.text
    return text[:limit]


class MinerUClient:
    """MinerU API 客户端: 批量上传 PDF → 轮询解析 → 下载结果。"""

    DEFAULT_API_URL = "https://mineru.net/api/v4/file-urls/batch"
    RESULT_URL_TEMPLATE = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        authorization: str = "",
        model_version: str = "vlm",
        output_dir: str = "mineru_result",
        poll_max_retries: int = 120,
        poll_interval: int = 5,
    ) -> None:
        self.api_url = api_url
        # 防御性清洗: 去掉首尾空白/换行 (常见于从 .env / 配置粘贴带入的 \r\n),
        # 否则 requests 会因 header 含回车符直接拒绝请求。
        self.authorization = (authorization or "").strip()
        self.model_version = model_version
        self.output_dir = output_dir
        self.poll_max_retries = poll_max_retries
        self.poll_interval = poll_interval

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": self.authorization,
        }

    def submit_batch(self, file_paths: List[str]) -> Dict[str, Any]:
        """提交批量解析任务, 上传文件, 返回 batch_id + urls。"""
        file_items = []
        for fp in file_paths:
            name = os.path.basename(fp)
            data_id = os.path.splitext(name)[0]
            file_items.append({"name": name, "data_id": data_id})

        data = {"files": file_items, "model_version": self.model_version}

        try:
            resp = requests.post(self.api_url, headers=self._headers(), json=data)
        except Exception as e:
            raise RuntimeError(f"MinerU API 请求失败: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"MinerU API HTTP {resp.status_code}: {_safe_resp_text(resp)}"
            )

        result = resp.json()
        if result.get("code") != 0:
            raise RuntimeError(f"MinerU API 错误: {result.get('msg')}")

        batch_id = result["data"]["batch_id"]
        urls = result["data"]["file_urls"]
        logger.info(f"batch_id={batch_id}, 获得 {len(urls)} 个上传 URL")
        return {"batch_id": batch_id, "urls": urls}

    def upload_files(
        self, file_paths: List[str], urls: List[str],
    ) -> int:
        """将文件上传到 MinerU 提供的预签名 URL。

        返回成功上传的文件数。
        """
        ok = 0
        for i, url in enumerate(urls):
            if i >= len(file_paths):
                break
            try:
                with open(file_paths[i], "rb") as f:
                    up_resp = requests.put(url, data=f)
                if up_resp.status_code == 200:
                    logger.info(f"上传成功: {file_paths[i]}")
                    ok += 1
                else:
                    logger.warning(
                        f"上传失败: {file_paths[i]} HTTP {up_resp.status_code}"
                    )
            except Exception as e:
                logger.warning(f"上传异常: {file_paths[i]} - {e}")
        return ok

    def poll_batch_result(
        self,
        batch_id: str,
        max_retries: Optional[int] = None,
        interval: Optional[int] = None,
        timeout_remaining: Optional[int] = None,
    ) -> Optional[Dict]:
        """轮询查询批处理任务的解析结果。

        完成状态为 done, 失败状态为 failed。

        Args:
            batch_id: 批处理 ID
            max_retries: 最大轮询次数
            interval: 轮询间隔秒数
            timeout_remaining: 剩余超时秒数, 到期即返回 None
        """
        max_retries = max_retries or self.poll_max_retries
        interval = interval or self.poll_interval
        result_url = self.RESULT_URL_TEMPLATE.format(batch_id=batch_id)
        t0 = time.time()

        for attempt in range(max_retries):
            if timeout_remaining and (time.time() - t0) >= timeout_remaining:
                logger.error(f"轮询超时 (剩余 {timeout_remaining}s 已耗尽)")
                return None

            try:
                resp = requests.get(result_url, headers=self._headers())
                if resp.status_code != 200:
                    logger.info(f"[{attempt + 1}] 查询失败 status: {resp.status_code}")
                    time.sleep(interval)
                    continue

                res = resp.json()
                if res.get("code") != 0:
                    logger.info(f"[{attempt + 1}] 查询返回错误: {res.get('msg')}")
                    time.sleep(interval)
                    continue

                extract_results = res["data"].get("extract_result", [])
                if not extract_results:
                    logger.info(f"[{attempt + 1}] 暂无结果, 继续等待...")
                    time.sleep(interval)
                    continue

                all_done = True
                for item in extract_results:
                    state = item.get("state")
                    file_name = item.get("file_name")
                    logger.info(f"[{attempt + 1}] {file_name} 状态: {state}")
                    if state == "failed":
                        logger.error(f"{file_name} 解析失败: {item.get('err_msg')}")
                        return None
                    if state != "done":
                        all_done = False

                if all_done:
                    return res["data"]
                time.sleep(interval)
            except Exception as e:
                logger.debug(f"[{attempt + 1}] 查询异常: {e}")
                time.sleep(interval)

        logger.error("超过最大轮询次数, 任务仍未完成")
        return None

    def save_extract_result(self, extract_data: Dict) -> None:
        """下载并保存解析结果的 zip 包。

        每个文件解析结果会以 zip 包形式提供, 包含 *_content_list.json,
        *_middle.json, *.md 等内容。下载 zip 并提取其中的 json 文件保存。
        """
        os.makedirs(self.output_dir, exist_ok=True)

        for item in extract_data.get("extract_result", []):
            file_name = item.get("file_name")
            zip_url = item.get("full_zip_url")
            if not zip_url:
                logger.warning(f"{file_name} 没有可下载的结果包")
                continue

            base_name = os.path.splitext(file_name)[0]
            sub_dir = os.path.join(self.output_dir, base_name)
            os.makedirs(sub_dir, exist_ok=True)

            logger.info(f"正在下载 {file_name} 解析结果...")
            zip_resp = requests.get(zip_url)
            if zip_resp.status_code != 200:
                logger.warning(
                    f"{file_name} 解析结果下载失败 HTTP {zip_resp.status_code}"
                )
                continue

            with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
                zf.extractall(sub_dir)
                logger.info(f"{file_name} 已解压到 {sub_dir}")

                for inner_name in zf.namelist():
                    if inner_name.endswith(".json"):
                        json_path = os.path.join(sub_dir, inner_name)
                        try:
                            with open(json_path, "r", encoding="utf-8") as jf:
                                parsed = json.load(jf)
                            with open(json_path, "w", encoding="utf-8") as jf:
                                json.dump(parsed, jf, ensure_ascii=False, indent=2)
                        except Exception as e:
                            logger.warning(f"处理 {json_path} 失败: {e}")

    def process(
        self, file_paths: List[str], save_batch_json: bool = True, timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """完整流程: 提交 → 上传 → 轮询 → 保存。

        Args:
            file_paths: PDF 文件路径列表
            save_batch_json: 是否保存批处理 JSON
            timeout: 整体超时秒数, 超过则抛出 TimeoutError (None 则不限时)

        Returns:
            {"batch_id": str, "output_dir": str, "batch_json": str | None}

        Raises:
            TimeoutError: 超过 timeout 秒未完成
        """
        t_start = time.time()

        def _check_timeout() -> None:
            if timeout and (time.time() - t_start) > timeout:
                raise TimeoutError(f"MinerU 解析超过 {timeout}s 超时")

        # 1) 提交批量任务
        _check_timeout()
        batch_info = self.submit_batch(file_paths)
        batch_id = batch_info["batch_id"]
        urls = batch_info["urls"]

        # 2) 上传文件
        _check_timeout()
        uploaded = self.upload_files(file_paths, urls)
        if uploaded == 0:
            raise RuntimeError(
                f"所有文件上传失败 ({len(file_paths)} 个), 请检查文件路径是否正确。"
                f"注意: 路径必须是文件, 不能是目录。"
            )
        if uploaded < len(file_paths):
            logger.warning(f"部分文件上传失败: {uploaded}/{len(file_paths)} 成功")

        # 3) 轮询结果 (带剩余超时计算)
        logger.info("等待 MinerU 解析完成...")
        remaining = None
        if timeout:
            remaining = max(1, int(timeout - (time.time() - t_start)))
        extract_data = self.poll_batch_result(batch_id, timeout_remaining=remaining)
        if extract_data is None:
            raise TimeoutError(f"MinerU 解析超过 {timeout}s 超时")

        # 4) 保存批处理结果
        _check_timeout()
        batch_json_path = None
        if save_batch_json:
            os.makedirs(self.output_dir, exist_ok=True)
            batch_json_path = os.path.join(self.output_dir, f"batch_{batch_id}.json")
            with open(batch_json_path, "w", encoding="utf-8") as f:
                json.dump(extract_data, f, ensure_ascii=False, indent=2)
            logger.info(f"批处理结果已保存: {batch_json_path}")

        # 5) 下载并解压解析结果
        self.save_extract_result(extract_data)

        return {
            "batch_id": batch_id,
            "output_dir": self.output_dir,
            "batch_json": batch_json_path,
        }
