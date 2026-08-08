import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


def train(model, optim, loss_fn, dataloader, test_dataloader, device, images, blocks, learning_rate, epochs, batch_size, index_path):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    #Training Loop
    for epoch in range(epochs):
        print(f"Epoch: ({epoch})")

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optim.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim = 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        
        train_acc = correct / total
        train_loss = total_loss / len(dataloader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        #Testing
        model.eval()
        predictions = []
        true_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in test_dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                true_labels.extend(batch_y.cpu().numpy())
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, dim = 1)
                predictions.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        val_loss /= len(test_dataloader)
        val_f1 = f1_score(true_labels, predictions, average = 'macro')
        val_prec = precision_score(true_labels, predictions, average = 'macro', zero_division = 0)
        val_rec = recall_score(true_labels, predictions, average = 'macro')
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)
        print(f"Train Loss: {train_loss}, Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}, Val Acc: {accuracy}, Val Prec: {val_prec}, Val_recall: {val_rec}, Val F1 score: {val_f1}")
        print("-" * 100)
    
    #Saving the weights of the trained model
    torch.save(model.state_dict(), f'{index_path}/model.pth')

    #Printing accuracy
    print(f"Accuracy: {accuracy_score(true_labels, predictions)}")

    epochs_range = range(epochs)

    plt.figure(figsize = (12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label = "Training Loss", color = "blue")
    plt.plot(epochs_range, val_losses, label = "Validation Loss", color = "red")
    plt.title("Loss per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label = "Training Accuracy", color = "blue")
    plt.plot(epochs_range, val_accuracies, label = "Validation Accuracy", color = "red")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend

    plt.tight_layout()
    plt.savefig("curves.png")
    plt.show()