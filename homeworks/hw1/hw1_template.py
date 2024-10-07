import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix


# Для установки рандомного состояния
def set_seed(seed=42):
    np.random.seed(seed)  # Для NumPy
    torch.manual_seed(seed)  # Для PyTorch на CPU

set_seed()


# Класс для нейронной сети
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(360, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 3)

    # Для обучения
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

    # Для предикта
    def predict(self, x):
        with torch.no_grad():
            # Прогоняем по слоям
            x = self.forward(x) 

            # Получаем вероятности 
            probabilities = torch.softmax(x, dim=1)

            # Переводим в классы  
            predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes 
    
# Загружаем данные
def load_data(train_csv, val_csv, test_csv):
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    # Тренировочный датасет
    X_train = df_train.drop(["order0", "order1", "order2"], axis=1)
    y_train = df_train["order0"]

    # Валидационный датасет
    X_val = df_val.drop(["order0", "order1", "order2"], axis=1)
    y_val = df_val["order0"]

    # Тестовый датасет
    X_test = pd.read_csv(test_csv)

    return X_train, y_train, X_val, y_val, X_test


# Инициализирем модель
def init_model(learning_rate):
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    return model, criterion, optimizer


# Предикт модели на валидации или обучении
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        predictions = model.predict(X).numpy()
        y = y.numpy()

        accuracy = accuracy_score(y_true=y, y_pred=predictions)
        conf_matrix = confusion_matrix(y_true=y, y_pred=predictions)

    return predictions, accuracy, conf_matrix


# Обучение модели
def train(model, criterion, optimizer, X_train, y_train, epochs, batch_size):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            # Обновляем градиенты
            optimizer.zero_grad()

            # Прогоняем по слоям (forward)
            outputs = model(X_batch).squeeze()

            # Считаем loss
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Обновляем параметры
            optimizer.step()
            train_loss += loss.item()

        train_loss /= X_train.size(0) // batch_size



def main(args):
   # Load data
    X_train, y_train, X_val, y_val, X_test = load_data(
        train_csv=args.train_csv, val_csv=args.val_csv, test_csv=args.test_csv
    )
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).long()

    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).long()

    X_test = torch.tensor(X_test.values, dtype=torch.float32)

    # Initialize model
    model, criterion, optimizer = init_model(args.lr)

    # Train model
    train(model, criterion, optimizer, X_train, y_train, args.num_epoches, args.batch_size)

    # Predict on val set
    predictions_val, accuracy_val, conf_matrix_val = evaluate(model, X_val, y_val)

    print(f"accuracy: {accuracy_val}")
    print(f"conf_matrix_val: {conf_matrix_val}")

    # Predict on test set
    predictions_test = model.predict(X_test).numpy()

    # Dump predictions to 'submission.csv'
    pd.Series(predictions_test).to_csv(args.out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', default='homeworks/hw1/data/train.csv')
    parser.add_argument('--val_csv', default='homeworks/hw1/data/val.csv')
    parser.add_argument('--test_csv', default='homeworks/hw1/data/test.csv')
    parser.add_argument('--out_csv', default='homeworks/hw1/submission.csv')
    parser.add_argument('--lr', default=6.106429040577424e-05)
    parser.add_argument('--batch_size', default=1024)
    parser.add_argument('--num_epoches', default=80)

    args = parser.parse_args()
    main(args)
