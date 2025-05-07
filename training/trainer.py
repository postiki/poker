import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import psutil
import gc
import time
from torch.utils.tensorboard import SummaryWriter

def get_optimal_batch_size():
    total_memory = psutil.virtual_memory().total
    available_memory = total_memory * 0.7
    image_size = 160 * 160 * 3 * 4
    return min(64, max(16, int(available_memory / (image_size * 2))))

def get_optimal_workers():
    cpu_count = psutil.cpu_count(logical=False)
    return min(4, max(2, cpu_count - 1))

def evaluate(model, val_loader, criterion, device='mps', use_amp=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 50 == 0:
                torch.mps.empty_cache()
                gc.collect()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='mps', use_amp=True):
    best_acc = 0.0
    scaler = torch.GradScaler() if use_amp else None
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 50 == 0:
                print(f'{50 / (time.time() - start_time):.2f} it/s')
                torch.mps.empty_cache()
                gc.collect()
                start_time = time.time()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'LR: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("LearningRate", scheduler.optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, 'best_card_classifier.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

    writer.close()