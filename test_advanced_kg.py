# test_advanced_kg.py
import os
import json
import argparse
import logging
import pickle
import random
from datetime import datetime
from pathlib import Path
import time
from typing import Optional, Dict, Any, List
from collections import defaultdict

from tqdm import tqdm

from memory_layer_kg import (
    AgenticMemorySystem,
    QASManager,
    QAItem,
)
from load_dataset import load_locomo_dataset  # LoComo 数据集
from utils import calculate_metrics, aggregate_metrics
import sys


# ====================== 日志相关 ======================

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("advanced_kg_eval")
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ====================== Agent 类 ======================

class AdvancedKGAgent:
    """
    用对话 turn 构建经验簇 + 知识图谱，并用经验回答 QA 的 Agent。

    核心点：
    - 从 turn.speaker / turn.text 构造 QAItem（answer 为空字符串）
    - 用 AgenticMemorySystem 的聚类 + 经验抽取 + KG 抽取能力
    - 用图统一检索（graph_retrieve_passages）作为上下文
    """

    def __init__(
        self,
        model: str,
        backend: str,
        temperature_c5: float = 0.5,
        retrieve_k: int = 10,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        emb_model_name: str = "all-MiniLM-L6-v2",
        use_clusters: bool = True,   # 是否使用经验簇做上下文 / 构建经验子图
        use_kg: bool = True,         # 是否使用 KG 做上下文
    ):
        self.memory_system = AgenticMemorySystem(
            model_name=emb_model_name,
            llm_backend=backend,
            llm_model=model,
            api_key=None,
            api_base=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        # 用来把 turn 转成 QAItem（算 embedding）
        self.qa_manager = QASManager(emb_model_name)
        self.turn_qas: List[QAItem] = []

        self.temperature_c5 = temperature_c5
        self.retrieve_k = retrieve_k

        self.use_clusters = use_clusters
        self.use_kg = use_kg

    # ---------- 构建知识阶段 ----------

    def add_turn(self, speaker: str, text: str, time_str: Optional[str] = None):
        """
        把一个对话 turn 加入为“伪 QAItem”，用于聚类/经验/KG 抽取。
        question = 原始文本，answer 为空字符串。
        """
        question = text
        qa_item = self.qa_manager.create(
            question=question,
            answer="",
            speaker=speaker,
            time_str=time_str
        )
        self.turn_qas.append(qa_item)
        return qa_item

    def build_knowledge(
        self,
        eps: float = 0.3,
        min_samples: int = 2,
        max_recluster_rounds: int = 2,
        window_size: int = 4,
        qa_items: Optional[List[QAItem]] = None,
        session_id: Optional[str] = None,
        build_clusters: bool = True,   # 控制是否构建/更新经验簇
        build_kg: bool = True,         # 控制是否构建/更新 KG
    ):
        """
        根据当前收集到的 QAItems：
        - 抽取经验簇（cluster + experiences）
        - 抽取知识图谱（entities + relations）

        通过 build_clusters / build_kg 控制只构建哪一部分。
        """
        target_qas = qa_items
        if not target_qas:
            return

        # 1) 抽取经验簇（聚类 + 一致性 + 经验抽取）
        if build_clusters:
            self.memory_system.incremental_experiences_from_qas(
                qa_items=target_qas,
                window_size=window_size,
                sim_high=0.8,
                sim_low=0.5,
                eps=eps,
                min_samples=min_samples,
                max_recluster_rounds=max_recluster_rounds,
                update_exp_threshold=5,
            )

        # 2) 抽取 KG（滑动窗口）
        if build_kg:
            self.memory_system.extract_kg_from_qa_windows(
                qa_items=target_qas,
                session_id=session_id,
            )
            # 抽取完 KG 后重建一次 triple 索引，用于后续检索
            self.memory_system.build_triple_index()

    # ---------- 利用经验 + KG 回答问题 ----------

    def _build_full_context(
        self,
        question: str,
        top_n_clusters: int = 3,   # clusters_only 模式下还会用到
        top_k_exps: int = 10,
        kg_max_entities: int = 3,  # 保留参数以防后面扩展
        kg_max_hop: int = 1,
        top_k_triples: int = 6,  # 保留参数以防后面扩展
        max_experiences = 6
    ) -> tuple[str, str, str]:
        """
        统一检索版本：

        - use_kg=True 时：
          使用 graph_retrieve_passages，在图上同时检索：
            * triples（结构化 KG）
            * passages / experiences（文本证据）
        - use_kg=False 且 use_clusters=True 时：
          使用旧的 retrieve_experiences_for_question，仅用经验簇文本
        """

        exp_context = ""
        kg_context = ""

        # ✅ 推荐默认：use_kg=True，统一从图检索（经验子图挂在 KG 里）
        if self.use_kg:
            graph_kg_context, graph_text_context = (
                self.memory_system.graph_retrieve_passages(
                    question=question,
                    top_k_triples=top_k_triples,
                    max_hop=kg_max_hop,
                    max_passages=6,
                    max_experiences=max_experiences,
                )
            )

            exp_context = graph_text_context or ""
            kg_context = graph_kg_context or ""

        # ✅ clusters_only 模式：纯经验检索，不依赖 KG
        elif self.use_clusters:
            exps = self.memory_system.retrieve_experiences_for_question(
                question=question,
                top_n_clusters=top_n_clusters,
                top_k_exps=top_k_exps,
            )
            if exps:
                exp_context = "\n".join([f"[{e.type}] {e.content}" for e in exps])

        # 拼装成 full_context
        parts = []
        if exp_context:
            parts.append("=== TEXT EVIDENCE (PASSAGES & EXPERIENCES) ===")
            parts.append(exp_context)
        if kg_context:
            parts.append("=== KNOWLEDGE GRAPH TRIPLES ===")
            parts.append(kg_context)

        if not parts:
            full_context = "No explicit past experiences or KG facts found."
        else:
            full_context = "\n".join(parts)

        return full_context, exp_context, kg_context

    def answer_question(self, question: str, category: int, golden_answer: str,top_k_triples: int =6, max_experiences = 6) -> tuple[str, str, str, str]:
        """
        Answer LoComo QA using experiences + KG with improved prompt structure.
        """

        assert category in [1, 2, 3, 4, 5]
        
        full_context, retrieved_experience, retrieved_knowledge= self._build_full_context(
            question=question,
            top_n_clusters=3,
            top_k_exps=self.retrieve_k,
            kg_max_entities=3,
            kg_max_hop=1,
            top_k_triples=top_k_triples,
            max_experiences = max_experiences
        )

        if not full_context or not str(full_context).strip():
            retrieved_experience = retrieved_experience or ""
            retrieved_knowledge = retrieved_knowledge or ""

        # 说明段落
        shared_intro = """
Answer using only the supported information from the context.
If the answer is unclear, respond briefly without adding unsupported details.
        """.strip()

        # 上下文块
        context_block = f"""
--- context start ---
{full_context}
--- context end ---
        """.strip()

        temperature = 0.7

        if category == 5:
            # 二选一
            options = []
            if random.random() < 0.5:
                options = ["Not mentioned in the conversation", golden_answer]
            else:
                options = [golden_answer, "Not mentioned in the conversation"]

            user_prompt = f"""
{shared_intro}

{context_block}

Question:
{question}

Choose the more appropriate of the following two answers, based only on the information above:

1) {options[0]}
2) {options[1]}

Return only the chosen answer.
            """.strip()

            temperature = 0.01

        elif category == 2:
            # 时间类问题
            user_prompt = f"""
{shared_intro}

{context_block}

Use DATE of CONVERSATION to answer with an approximate date. 
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects. 

Question:
{question}
Short Answer:
            """.strip()

        elif category == 3:
        
            user_prompt = f"""
{shared_intro}

{context_block}

Answer using only a short phrase.

If the question is about likelihood or hypotheticals, start with  one of:
"Likely yes", "Likely no", "Yes", or "No".

Then add at most a few words based on the context.Do NOT explain uncertainty or say things like "not explicitly stated" or "cannot be sure".

Question:
{question}
Short Answer:
            """.strip()

        else:
            # 默认普通问答
            user_prompt = f"""
{shared_intro}

{context_block}

Answer the question with the SHORTEST possible phrase from the context.
If the answer is present, COPY the words from the context.
Do NOT add new information or explanations.


Question:
{question}
Short Answer:
            """.strip()

        # JSON 输出要求
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        resp = self.memory_system.llm_controller.llm.get_completion(
            prompt=user_prompt,
            response_format=response_format,
            temperature=temperature,
        )
        try:
            response_cleaned = resp[resp.find('{'):resp.rfind('}')+1]
            data = json.loads(response_cleaned)
            ans = data.get("answer", "")
            return str(ans), retrieved_experience, retrieved_knowledge, user_prompt
        except Exception:
            return str(resp), retrieved_experience, retrieved_knowledge, user_prompt


# ====================== KG 导出工具 ======================

def kg_to_json(kg) -> Dict[str, Any]:
    entities = []
    for ent in kg.entities.values():
        entities.append(
            {
                "id": ent.id,
                "name": ent.name,
                "metadata": ent.metadata,
            }
        )

    relations = []
    for rel in kg.relations.values():
        relations.append(
            {
                "id": rel.id,
                "source": rel.source,
                "target": rel.target,
                "relation_type": rel.relation_type,
                "metadata": rel.metadata,
            }
        )

    return {"entities": entities, "relations": relations}


# ====================== 主评估逻辑 ======================

def evaluate_dataset(
    dataset_path: str,
    model: str,
    output_path: Optional[str],
    ratio: float,
    backend: str,
    temperature_c5: float,
    retrieve_k: int,
    sglang_host: str,
    sglang_port: int,
    eps: float = 0.3,
    min_samples: int = 2,
    max_recluster_rounds: int = 2,
    window_size: int = 12,
    save_detailed_logs: bool = True,
    detailed_logs_path: Optional[str] = None,
    use_clusters: bool = True,   # 是否用 clusters 作为上下文 / 构建经验子图
    use_kg: bool = True,         # 是否用 KG 作为上下文
    allow_categories: Optional[List[int]] = None,
    cache_dir: Optional[str] = None,
    top_k_triples: int = 6,
    max_experiences = 6
):
    """
    整体流程：
    1. 按样本遍历 LoComo 数据
    2. 每个 sample：
       - 始终从原始对话重新构建 QAItem（不使用 QA 缓存）
       - 如果 clusters / KG 有缓存就加载，没有就重建（缺啥补啥）
       - 最后如果两边都启用，就统一用 clusters 重建经验子图挂到 KG
       - 根据 use_clusters / use_kg 决定用哪一种 memory 来回答
    3. 用 sample.qa 做问题评估
    4. 计算指标 & 输出结果
    """
    datalabel = dataset_path.split('/')[-1].split('.')[0]
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    base_dir = os.path.dirname(__file__)

    




    # 日志
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"test_advanced_kg_{model}_{backend}_ratio{ratio}_{timestamp}.log",
    )
    logger = setup_logger(log_file)

    # 打印上下文模式
    logger.info(f"Context config: use_clusters={use_clusters}, use_kg={use_kg}")

    # 详细对话记录保存路径
    if save_detailed_logs:
        if detailed_logs_path is None:
            detailed_logs_path = os.path.join(
                base_dir,
                "detailed_logs",
                f"KGmem_{model}_{datalabel}.json"
            )
        os.makedirs(os.path.dirname(detailed_logs_path), exist_ok=True)
        logger.info(f"Detailed QA logs will be saved to: {detailed_logs_path}")

    logger.info(f"Loading dataset from {dataset_path}")
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")

    # 按 ratio 只取前一部分样本
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples (ratio={ratio:.2f})")

    if cache_dir is None:
        cache_dir = os.path.join(base_dir, f"cached_advanced_kg_{backend}_{model}")
    else:
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(base_dir, cache_dir)

    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Cache directory: {cache_dir}")

    # 评估统计
    if allow_categories is None:
        allow_categories = [1, 2, 3, 4, 5]
    total_questions = 0
    category_counts = defaultdict(int)
    error_num = 0
    all_metrics = []
    all_categories = []
    results = []

    # 整体 KG 统计
    total_entities = 0
    total_relations = 0

    detailed_records = []

    for sample_idx, sample in enumerate(samples):
        logger.info(f"\n===== Processing sample {sample_idx + 1}/{len(samples)} =====")
        agent = AdvancedKGAgent(
            model=model,
            backend=backend,
            temperature_c5=temperature_c5,
            retrieve_k=retrieve_k,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            use_clusters=use_clusters,
            use_kg=use_kg,
        )


        allow = set(allow_categories or [1,2,3,4,5])
        sample_total_qas = sum(1 for _qa in sample.qa if int(_qa.category) in allow)

        logger.info(
            f"[SAMPLE {sample_idx+1}/{len(samples)}] START answering QA "
            f"(total_qas={sample_total_qas}, allow={sorted(list(allow))})"
        )

        sample_done = 0



        # 缓存文件名（仅 clusters 和 KG）
        clusters_cache_file = os.path.join(
            cache_dir,
            f"clusters_sample_{sample_idx}.pkl",
        )
        kg_cache_file = os.path.join(
            cache_dir,
            f"kg_sample_{sample_idx}.pkl",
        )

        # ---------- 1. 始终重建 QAItem（从 sample.conversation.sessions） ----------
        agent.turn_qas = []
        session_qas_map: Dict[str, List[QAItem]] = {}

        for session_id, session in sample.conversation.sessions.items():
            t_time = session.date_time
            logger.info(f"Processing session_id: {session_id}, date_time: {t_time}")
            session_qas: List[QAItem] = []
            for turn in session.turns:
                qa_item = agent.add_turn(turn.speaker, turn.text, time_str=str(t_time))
                session_qas.append(qa_item)
            session_qas_map[session_id] = session_qas

        # ---------- 2. clusters / KG 按需加载或重建（缺啥补啥） ----------

        # 2.1 经验簇 / 经验 cache
        if agent.use_clusters:
            has_clusters = os.path.exists(clusters_cache_file)
            if has_clusters:
                logger.info(f"Loading cached clusters for sample {sample_idx}")
                try:
                    with open(clusters_cache_file, "rb") as f:
                        clusters = pickle.load(f)
                    agent.memory_system._clusters = clusters
                    logger.info(f"Loaded clusters: {len(agent.memory_system._clusters)}")
                except Exception as e:
                    logger.info(f"Error loading clusters cache: {e}")
                    has_clusters = False
        else:
            has_clusters = False

        # 2.2 KG（注意：这里只保存“基础知识图”，不包含经验子图）
        if agent.use_kg:
            has_kg = os.path.exists(kg_cache_file)
            if has_kg:
                logger.info(f"Loading cached KG for sample {sample_idx}")
                try:
                    with open(kg_cache_file, "rb") as f:
                        kg_obj = pickle.load(f)
                    agent.memory_system.kg = kg_obj
                    # 从缓存加载 KG 后，需要重建 triple 索引供 graph 检索使用
                    agent.memory_system.build_triple_index()
                    logger.info(
                        f"Loaded KG: entities={len(agent.memory_system.kg.entities)}, "
                        f"relations={len(agent.memory_system.kg.relations)}"
                    )
                except Exception as e:
                    logger.info(f"Error loading KG cache: {e}")
                    has_kg = False
        else:
            has_kg = False

        # 2.3 根据 missing 状态决定需要构建哪一部分（只对启用的部分生效）
        need_clusters = agent.use_clusters and (not has_clusters)
        need_kg = agent.use_kg and (not has_kg)

        if need_clusters or need_kg:
            logger.info(
                f"Rebuilding missing parts for sample {sample_idx} "
                f"(clusters_missing={need_clusters}, kg_missing={need_kg})"
            )
            for session_id, session_qas in session_qas_map.items():
                agent.build_knowledge(
                    eps=eps,
                    min_samples=min_samples,
                    max_recluster_rounds=max_recluster_rounds,
                    window_size=window_size,
                    qa_items=session_qas,
                    session_id=session_id,
                    build_clusters=need_clusters,
                    build_kg=need_kg,
                )

            # 2.4 保存刚构建的缓存（哪个缺就存哪个）
            if need_clusters:
                try:
                    with open(clusters_cache_file, "wb") as f:
                        pickle.dump(agent.memory_system._clusters, f)
                    logger.info(f"Saved clusters cache for sample {sample_idx}")
                except Exception as e:
                    logger.info(f"Error saving clusters cache: {e}")

            if need_kg:
                try:
                    with open(kg_cache_file, "wb") as f:
                        pickle.dump(agent.memory_system.kg, f)
                    logger.info(f"Saved KG cache for sample {sample_idx}")
                except Exception as e:
                    logger.info(f"Error saving KG cache: {e}")


        if (
            agent.use_clusters
            and agent.use_kg
            and getattr(agent.memory_system, "_clusters", None)
            and getattr(agent.memory_system, "kg", None)
        ):
            logger.info("Rebuilding experience subgraph from clusters (final)...")
            agent.memory_system.rebuild_experience_subgraph_from_clusters()

        all_qas: List[QAItem] = []
        for _sid, _qas in session_qas_map.items():
            all_qas.extend(_qas)

        agent.memory_system.rebuild_retriever_from_qas(all_qas)
        logger.info(f"Rebuilt global passage retriever from QAItems: {len(all_qas)}")

        # 更新 KG 统计（KG 对象在 AgenticMemorySystem 初始化时就存在，只是可能为空）
        total_entities += len(agent.memory_system.kg.entities)
        total_relations += len(agent.memory_system.kg.relations)

        # ---------- 3. 用 QA 做评估 ----------
        for qa in sample.qa:
            if int(qa.category) not in allow_categories:
                continue

            total_questions += 1
            category_counts[qa.category] += 1

            prediction_raw, retrieved_experience, retrieved_knowledge, prompt = agent.answer_question(
                qa.question,
                qa.category,
                qa.final_answer,
                top_k_triples=top_k_triples,
                max_experiences = max_experiences
            )

            prediction = str(prediction_raw)

            if qa.category == 5:
                reference = "Not mentioned in the conversation"
            else:
                reference = qa.final_answer
            
            logger.info(f"\nQuestion {total_questions}: {qa.question}")
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Reference: {reference}")
            logger.info(f"Category: {qa.category}")
            logger.info(f"Retrieved Experiences: {retrieved_experience}")
            logger.info(f"Retrieved Knowledge: {retrieved_knowledge}")

            # 计算指标
            if reference:
                metrics = calculate_metrics(prediction, reference)
            else:
                metrics = {
                    "exact_match": 0,
                    "f1": 0.0,
                    "rouge1_f": 0.0,
                    "rouge2_f": 0.0,
                    "rougeL_f": 0.0,
                    "bleu1": 0.0,
                    "bleu2": 0.0,
                    "bleu3": 0.0,
                    "bleu4": 0.0,
                    "bert_f1": 0.0,
                    "meteor": 0.0,
                    "sbert_similarity": 0.0,
                }

            all_metrics.append(metrics)
            all_categories.append(qa.category)

            # 构建简要结果
            result = {
                "sample_id": sample_idx,
                "question": qa.question,
                "prediction": prediction,
                "reference": reference,
                "category": qa.category,
                "metrics": metrics
            }
            results.append(result)

            # 详细记录
            detailed_record = {
                "sample_id": sample_idx,
                "question_id": total_questions,
                "question": qa.question,
                "category": qa.category,
                "prediction": prediction,
                "reference": reference,
                "retrieved_experiences": retrieved_experience,
                "retrieved_knowledge": retrieved_knowledge,
                "prompt":prompt,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
            detailed_records.append(detailed_record)

            if total_questions % 10 == 0:
                logger.info(f"Processed {total_questions} questions so far.")

            sample_done += 1
            logger.info(
                f"[SAMPLE {sample_idx+1}/{len(samples)}] QA progress: {sample_done}/{sample_total_qas}"
            )

            sys.stdout.flush()

    # ---------- 4. 保存详细日志到 JSON 文件 ----------
    if save_detailed_logs and detailed_records:
        try:
            with open(detailed_logs_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "detailed_records": detailed_records
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Detailed QA logs saved to: {detailed_logs_path}")
        except Exception as e:
            logger.error(f"Failed to save detailed logs: {e}")

    # ---------- 5. 汇总指标 & 导出结果 ----------

    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    final_results = {
        "model": model,
        "dataset": dataset_path,
        "total_questions": total_questions,
        "context_config": {
            "use_clusters": use_clusters,
            "use_kg": use_kg,
        },
        "category_distribution": {str(cat): count for cat, count in category_counts.items()},
        "aggregate_metrics": aggregate_results,
        "individual_results": results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        logger.info(f"\nResults saved to {output_path}")

    logger.info("\nEvaluation Summary:")
    logger.info(f"Total questions evaluated: {total_questions}")
    logger.info(f"Error parses: {error_num}")
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"Category {category}: {count} ({count/total_questions*100:.1f}%)")

    logger.info("\nAggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                logger.info(f"    {stat_name}: {value:.4f}")

    logger.info(f"\nGlobal KG stats: entities={total_entities}, relations={total_relations}")

    if save_detailed_logs:
        logger.info(f"\nDetailed QA logs saved to: {detailed_logs_path}")

    logger.info("\nDone.")

    return final_results


# ====================== main：薄壳风格 ======================

def main():
    parser = argparse.ArgumentParser(description="Evaluate advanced KG-based agent on LoComo dataset")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b",
                        help="Model name for LLM backend")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results (JSON)")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="ollama",
                        help="Backend to use (openai, ollama, or sglang)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for category-5 adversarial questions")
    parser.add_argument("--retrieve_k", type=int, default=5,
                        help="How many experience items to use as context")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")
    parser.add_argument("--save_detailed_logs", action="store_true", default=True,
                        help="Save detailed QA logs with questions, contexts, and answers")
    parser.add_argument("--detailed_logs_path", type=str, default=None,
                        help="Path to save detailed QA logs (JSON format)")
    parser.add_argument("--top_k_triples", type=int, default=6,
                        help="Number of KG triples to retrieve as context")
    parser.add_argument("--max_experiences", type=int, default=6,
                        help="Maximum number of experiences to retrieve from KG")
    # 新增：控制只用 KG、只用 clusters 或两者都用
    parser.add_argument(
        "--context_mode",
        type=str,
        default="both",
        choices=["both", "kg_only", "clusters_only"],
        help="Which memory context to use for answering: both, kg_only, or clusters_only",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="要评估的类别，例如: 1,3,5；不填则默认评估所有 1-5 类",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cached_advanced_kg_openai_deepseek-chat",
        help="Directory for per-sample KG/clusters cache (reuse existing cache if provided)",
    )

    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, args.dataset)

    # 解析 context_mode -> use_clusters / use_kg
    if args.context_mode == "both":
        use_clusters = True
        use_kg = True
    elif args.context_mode == "kg_only":
        use_clusters = False
        use_kg = True
    else:  # "clusters_only"
        use_clusters = True
        use_kg = False

    if args.categories:
        allow_categories = [
            int(c.strip()) for c in args.categories.split(",") if c.strip()
        ]
    else:
        allow_categories = None  # 让 evaluate_dataset 自己用默认 [1..5]

    if args.output:
        output_path = os.path.join(os.path.dirname(__file__), args.output)
    else:
        output_path = f"output-KGmem/{args.dataset.split('/')[-1].split('.')[0]}/{args.model}.json"
        print(f"Output path: {output_path}")
    if args.detailed_logs_path:
        args.detailed_logs_path = os.path.join(os.path.dirname(__file__), args.detailed_logs_path)
    else:
        args.detailed_logs_path = detailed_logs_path = os.path.join(
                base_dir,
                "detailed_logs",
                f"KGmem_{args.model}_{dataset_path.split('/')[-1].split('.')[0]}.json"
            )

    evaluate_dataset(
        dataset_path=dataset_path,
        model=args.model,
        output_path=output_path,
        ratio=args.ratio,
        backend=args.backend,
        temperature_c5=args.temperature_c5,
        retrieve_k=args.retrieve_k,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
        save_detailed_logs=args.save_detailed_logs,
        detailed_logs_path=args.detailed_logs_path,
        use_clusters=use_clusters,
        use_kg=use_kg,
        allow_categories=allow_categories,
        cache_dir=args.cache_dir,
        top_k_triples=args.top_k_triples,
        max_experiences = args.max_experiences
    )


if __name__ == "__main__":
    main()
