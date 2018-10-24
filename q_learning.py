import gym
import numpy as np

env=gym.make("FrozenLake-v0")

env.render()

#q table oluşturma np.zeros((1,2))
q_table = np.zeros((env.observation_space.n,4))

oynama_sayısı=10000
y = 0.95  #gelecek önem katsayısı
öğrenme_katsayısı = 0.99
epsilon= 0.8  # random seçim için gerekli katsayı  ---->>>> hangi sıklıkla random seçim yapacağız
kaç_kere_oyun_bitti=0
for i in range(oynama_sayısı):
    durum = env.reset()
    
    oyun_bitti = False
    kaç_harekette_bitti=0
    while not oyun_bitti:
        kaç_harekette_bitti += 1
        
        if ((np.sum(q_table[durum,:]) == 0) or (np.random.rand(1)>epsilon)):
            
            a = np.random.randint(0, env.action_space.n)
        else:
           
            a = np.argmax(q_table[durum, :])
        yeni_durum, ödül, oyun_bitti, info = env.step(a)
        
        if(ödül==1):
            print(kaç_harekette_bitti)
            kaç_kere_oyun_bitti+=1
        ödül=ödül-0.2   # zaman ödülü
        q_table[durum, a] += ödül + öğrenme_katsayısı*(y*np.max(q_table[yeni_durum, :]) - q_table[durum, a])
        
        durum = yeni_durum
    
print(kaç_kere_oyun_bitti)    
        
