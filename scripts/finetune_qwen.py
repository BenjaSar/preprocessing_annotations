"""
Qwen2.5-VL Fine-tuning Infrastructure (Phase 3).

This module provides tools for preparing SFT (Supervised Fine-Tuning) data
and configuring LoRA fine-tuning for Qwen2.5-VL on floor plan room detection.

Usage:
    python finetune_qwen.py --prepare-data --input ./processed_annotations
    python finetune_qwen.py --train --model qwen/Qwen2.5-VL-7B --data ./finetune_data

This implements Phase 3 of the implementation roadmap:
  - Collect accumulated SFT training data from annotations pipeline
  - Format data for Qwen2.5-VL fine-tuning (instruction-following format)
  - Configure LoRA adapters to reduce memory footprint
  - Evaluate on held-out test set
"""

import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QwenFinetuneDataPreparator:
    """Prepare SFT data for Qwen2.5-VL fine-tuning."""
    
    def __init__(self, image_dir: Path, annotation_dir: Path, output_dir: Path):
        """Initialize data preparator.
        
        Args:
            image_dir: Directory containing floor plan images
            annotation_dir: Directory containing processed annotations (JSON)
            output_dir: Output directory for fine-tuning datasets
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_sft_data(self) -> Dict[str, int]:
        """
        Prepare SFT training dataset from annotations.
        
        Converts annotations to instruction-following format:
            Input: "Detect all rooms in this floor plan."
            Output: JSON with room detections
        
        Returns:
            Dict with split statistics: {train: N, val: N, test: N}
        """
        logger.info("Preparing SFT training data for Qwen2.5-VL...")
        
        # Load all annotations
        annotations = self._load_annotations()
        if not annotations:
            logger.warning("No annotations found")
            return {"train": 0, "val": 0, "test": 0}
        
        logger.info(f"Loaded {len(annotations)} annotations")
        
        # Convert to SFT format
        sft_examples = []
        for image_file, annotation in annotations.items():
            example = self._annotation_to_sft_example(image_file, annotation)
            if example:
                sft_examples.append(example)
        
        logger.info(f"Converted {len(sft_examples)} examples to SFT format")
        
        # Split into train/val/test (70/15/15)
        random.shuffle(sft_examples)
        n_total = len(sft_examples)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_data = sft_examples[:n_train]
        val_data = sft_examples[n_train:n_train + n_val]
        test_data = sft_examples[n_train + n_val:]
        
        # Save splits
        self._save_split(train_data, "train")
        self._save_split(val_data, "val")
        self._save_split(test_data, "test")
        
        logger.info(
            f"Saved splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
        )
        
        return {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        }
    
    def _load_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Load all annotation JSON files."""
        annotations = {}
        
        if not self.annotation_dir.exists():
            logger.warning(f"Annotation directory not found: {self.annotation_dir}")
            return annotations
        
        for json_file in self.annotation_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'rooms' in data:
                        annotations[json_file.stem] = data
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        return annotations
    
    def _annotation_to_sft_example(self, image_file: str, annotation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert annotation to SFT example format.
        
        Qwen2.5-VL expects instruction-following examples with images.
        
        F8: Include bbox data in SFT output to provide spatial grounding signals
        for improved room localization during fine-tuning.
        """
        rooms = annotation.get('rooms', [])
        if not rooms:
            return None
        
        # Build room output format with bbox data (F8)
        room_list = []
        for room in rooms:
            room_entry = {
                "room_name": room.get("name", room.get("room_name", "")),
                "room_type": room.get("type", room.get("category", "other")),
            }
            if room.get("room_number"):
                room_entry["room_number"] = room["room_number"]
            
            # F8: Add bbox data for spatial grounding
            bbox = room.get("bbox", [])
            if bbox and len(bbox) == 4:
                # Convert [x, y, w, h] to [x1, y1, x2, y2] if needed
                if bbox[2] > 0 and bbox[3] > 0 and (bbox[2] < bbox[0] or bbox[3] < bbox[1]):
                    # Likely already in [x1, y1, x2, y2] format
                    room_entry["bbox"] = {
                        "x1": float(bbox[0]),
                        "y1": float(bbox[1]),
                        "x2": float(bbox[2]),
                        "y2": float(bbox[3])
                    }
                else:
                    # Convert from [x, y, w, h]
                    room_entry["bbox"] = {
                        "x1": float(bbox[0]),
                        "y1": float(bbox[1]),
                        "x2": float(bbox[0] + bbox[2]),
                        "y2": float(bbox[1] + bbox[3])
                    }
            
            room_list.append(room_entry)
        
        # Get image dimensions for bbox normalization reference
        image_size = annotation.get("image_size", {})
        
        # Create SFT example with bbox data
        example = {
            "id": f"{image_file}",
            "instruction": "Analyze this floor plan image and identify all rooms. For each room, provide its name, number (if present), type, and bounding box coordinates. Return results as JSON.",
            "input": image_file,  # Will be image path at training time
            "output": json.dumps({
                "rooms": room_list,
                "total_rooms": len(room_list),
                "image_dimensions": image_size  # For bbox context
            }),
            "metadata": {
                "source": "preprocessing_annotations",
                "image_file": image_file,
                "num_rooms": len(room_list),
                "has_bboxes": all("bbox" in r for r in room_list)  # Track data completeness
            }
        }
        
        return example
    
    def _save_split(self, examples: List[Dict[str, Any]], split_name: str) -> None:
        """Save dataset split to JSONL format."""
        output_file = self.output_dir / f"{split_name}.jsonl"
        
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved {split_name} split to {output_file}")


class LoRAConfig:
    """LoRA configuration for Qwen2.5-VL fine-tuning."""
    
    # LoRA hyperparameters (recommended for 7B model on single GPU)
    r = 8  # LoRA rank
    lora_alpha = 16  # LoRA scaling factor
    lora_dropout = 0.05  # Dropout rate
    target_modules = ["q_proj", "v_proj"]  # Which modules to fine-tune
    
    # Training hyperparameters
    learning_rate = 1e-4
    num_train_epochs = 3
    per_device_train_batch_size = 2  # Adjust based on GPU memory
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 4
    
    # Other settings
    warmup_ratio = 0.1
    weight_decay = 0.0
    max_grad_norm = 1.0
    logging_steps = 100
    eval_steps = 500
    save_steps = 500
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            'r': cls.r,
            'lora_alpha': cls.lora_alpha,
            'lora_dropout': cls.lora_dropout,
            'target_modules': cls.target_modules,
            'learning_rate': cls.learning_rate,
            'num_train_epochs': cls.num_train_epochs,
            'per_device_train_batch_size': cls.per_device_train_batch_size,
            'per_device_eval_batch_size': cls.per_device_eval_batch_size,
            'gradient_accumulation_steps': cls.gradient_accumulation_steps,
            'warmup_ratio': cls.warmup_ratio,
            'weight_decay': cls.weight_decay,
            'max_grad_norm': cls.max_grad_norm,
            'logging_steps': cls.logging_steps,
            'eval_steps': cls.eval_steps,
            'save_steps': cls.save_steps,
        }


def generate_training_script(output_path: Path) -> None:
    """
    Generate a training script template for Qwen2.5-VL fine-tuning.
    
    This script is a reference implementation using transformers library.
    Actual training would use HuggingFace Trainer or similar.
    """
    script_content = '''#!/usr/bin/env python3
"""
Qwen2.5-VL LoRA Fine-tuning Script.

This script fine-tunes Qwen2.5-VL-7B on floor plan room detection using LoRA.

Requirements:
    pip install transformers peft torch bitsandbytes trl

Usage:
    python train_qwen_lora.py \\
        --model_id qwen/Qwen2.5-VL-7B \\
        --train_data finetune_data/train.jsonl \\
        --output_dir ./qwen_lora_checkpoint
"""

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import argparse
from pathlib import Path
import json

def load_dataset(data_file):
    """Load JSONL dataset."""
    examples = []
    with open(data_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='qwen/Qwen2.5-VL-7B')
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--eval_data', default=None)
    parser.add_argument('--output_dir', default='./qwen_lora_checkpoint')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Load datasets
    train_data = load_dataset(args.train_data)
    eval_data = load_dataset(args.eval_data) if args.eval_data else None
    
    # Configure training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.0,
        logging_steps=100,
        save_steps=500,
        eval_steps=500 if eval_data else None,
        evaluation_strategy="steps" if eval_data else "no",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        bf16=True,
        gradient_accumulation_steps=4,
    )
    
    # Create trainer (pseudocode - actual implementation would use custom dataset loader)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=processor,
        # packing=True,  # Optional: for better memory efficiency
    )
    
    # Train
    trainer.train()
    
    # Save final model
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\\nModel saved to {args.output_dir}")

if __name__ == '__main__':
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    output_path.chmod(0o755)
    logger.info(f"Generated training script template: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare and manage Qwen2.5-VL fine-tuning data"
    )
    parser.add_argument(
        '--prepare-data',
        action='store_true',
        help='Prepare SFT training data from annotations'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('./processed_annotations'),
        help='Input directory with processed annotations'
    )
    parser.add_argument(
        '--image-dir',
        type=Path,
        default=Path('./images'),
        help='Directory with floor plan images'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./finetune_data'),
        help='Output directory for fine-tuning data'
    )
    parser.add_argument(
        '--generate-script',
        action='store_true',
        help='Generate training script template'
    )
    parser.add_argument(
        '--lora-config',
        action='store_true',
        help='Print recommended LoRA configuration'
    )
    
    args = parser.parse_args()
    
    if args.prepare_data:
        logger.info("=" * 60)
        logger.info("PHASE 3: Fine-tuning Data Preparation")
        logger.info("=" * 60)
        
        preparator = QwenFinetuneDataPreparator(
            image_dir=args.image_dir,
            annotation_dir=args.input,
            output_dir=args.output
        )
        stats = preparator.prepare_sft_data()
        
        logger.info("\n" + "=" * 60)
        logger.info("Data Preparation Complete!")
        logger.info("=" * 60)
        logger.info(f"Train samples: {stats['train']}")
        logger.info(f"Val samples: {stats['val']}")
        logger.info(f"Test samples: {stats['test']}")
        logger.info(f"\nOutput directory: {args.output}")
        logger.info(f"Next step: Run training script on train.jsonl")
    
    if args.generate_script:
        script_path = args.output / "train_qwen_lora.py"
        generate_training_script(script_path)
        logger.info(f"\nTo train, run:")
        logger.info(f"  python {script_path} --train_data {args.output}/train.jsonl")
    
    if args.lora_config:
        logger.info("\n" + "=" * 60)
        logger.info("Recommended LoRA Configuration")
        logger.info("=" * 60)
        config = LoRAConfig.to_dict()
        for key, value in config.items():
            logger.info(f"{key:30s} = {value}")
        logger.info("\nNote: Adjust batch_size and gradient_accumulation_steps")
        logger.info("based on your GPU memory (adjust for A100 80GB, 24GB GPU, etc.)")


if __name__ == '__main__':
    main()
