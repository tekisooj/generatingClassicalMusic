import glob
import pickle
import numpy
import music21
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

def getNotes():
    notes = []


    for dirname, _, filenames in os.walk('/home/tekisooj/Desktop/projekatRI/chopinMidi/'):
        for filename in filenames:
    #parsiramo ulazne midi fajlove(imaju ext mid)
            midi = converter.parse(os.path.join(dirname, filename))

            print("Parsiramo %s" %filename)
            #u pocetku nemamo note za obradjivanje
            notesToParse = None
            #pokusavamo da razvrstamo po instrumentima 
            try:
                instruments = instrument.partitionByInstrument(midi)
                notesToParse = instruments.parts[0].recurse()
            except:
                notesToParse = midi.flat.notes

            #posto od tipova ulaza imamo note i akorde, moraju oba slucaja da se obrade
            #tj proveravamo sta je od ta dva u pitanju i dodajemo u []
            for el in notesToParse:
                if isinstance(el, note.Note):
                    notes.append(str(el.pitch))
              #akorde predstavljamo kao nota.nota.nota......
                if isinstance(el, chord.Chord):
                    notes.append('.'.join(str(i) for i in el.normalOrder))

            #serijalizujemo dobijene note i upisujemo u fajl da bismo kasnije mogli da
            #ih prenosimo i koristimo....
    with open('notes', 'wb') as fpath:
          pickle.dump(notes, fpath)

    print(len(notes))
    return notes   



notes = getNotes()



def write(output_notes):
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='test_output.mid')


def prepareSequences(notes, nDiff):

  #da bitmo predvideli koja nota/koji akord je na redu, koristimo prethodnih 100 
  ##########probati sa razl vrednostima
    sequenceLength = 100

  #zelimo da izdvojimo sve tonove koji su se javljali u nasim "uzorcima"
  #tj kompozicijama koje smo citali
    pitchNames = sorted(set(i for i in notes))


  #sada sve ucitane note zelimo da napravimo preslikavanje, tj da ih predstavimo
  #kao parove str, int 
  #to kasnije mozemo iskoristili da primenimo gradijentni spust(u lstm)
    noteToInt = dict((note, number) for number, note in enumerate(pitchNames))

    networkInput = []
    networkOutput = []

    n = len(notes)

  #izlaz ce biti prva nota ili akord koji dolaze nakon odgovarajuce ulazne 
  #sekvence 

    for i in range(0, n - sequenceLength):
        sequenceIn = notes[i:i + sequenceLength]
        sequenceOut = notes[i + sequenceLength]
        networkInput.append([noteToInt[note] for note in sequenceIn])
        networkOutput.append(noteToInt[sequenceOut])

    nPatterns = len(networkInput)

  # ulaz predstavljamo u formatu kompatibilnom sa lstm slojevima
    networkInput = numpy.reshape(networkInput, (nPatterns, sequenceLength, 1))
  # normalizujemo ulaz (delimo sa brojem razlicitih nota/akorda)
    networkInput = networkInput / float(nDiff)

    networkOutput = np_utils.to_categorical(networkOutput)

    return networkInput, networkOutput, pitchNames


def createNetwork(networkInput, nDiff):
    """ create the structure of the neural network """
    #LSTM(Long Short Trem Memory) je sloj RRN koja prima sekvencu ulaza i vraca
    #sekvencu ili matricu (u ovom slucaju sekvencu)
    #aktivacioni sloj odredjuje koju ce aktivacionu fju nasa mreza koristiti za
    #izdracunavanje 
     #za LSTM, Dense i Activation slojeve prvi parametar je broj cvorova u njima
    #dropout parametar predstavlja koliki deo ulaznih vrednosti ce biti odbacen
    #prilikom treniranja
    # input_shape daje do znjanja mrezi kakvog ce oblika biti podaci koje ce 
    #trenirati 


    ######treba se igrati malo sa ovim slojevima 
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(networkInput.shape[1], networkInput.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(nDiff))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    #poslednji sloj mreze ima isti broj cvorova kao nas izlaz da bi se direktno
    #mapiralo
    return model


def train(model, networkInput, networkOutput):

  #####ovaj deo treba objasniti prica je o gubicima
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacksList = [checkpoint]
    earlyStop = EarlyStopping(monitor = 'loss', verbose = 2, patience = 3, mode = 'min')
    model.fit(networkInput, networkOutput, epochs=200, batch_size=128, callbacks=callbacksList)
    
    
    
    return model


def trainNetwork():
    #u ovoj funkciji treniramo mrezu
    # broj svih tonova(bez duplikata)
    nDiff = len(set(notes))

    networkInput, networkOutput, pitchNames = prepareSequences(notes, nDiff)

    model = createNetwork(networkInput, nDiff)

    model = train(model, networkInput, networkOutput)

    return  model, networkInput, pitchNames, nDiff


def generateNotes(model, networkInput, pitchNames, nDiff):
  

  #od svih sekvenci tonova biramo jednu od koje cemo poceti
    start = numpy.random.randint(0, len(networkInput)-1)

    intToNote = dict((number, note) for number, note in enumerate(pitchNames))

    pattern = networkInput[start]

    predictionOutput = []

    for _ in range(500):
      
        predictionInput = numpy.reshape(pattern, (1, pattern.shape[0], 1))
        predictionInput = predictionInput/float(nDiff)

        prediction = model.predict(predictionInput, verbose = 0)

        index = numpy.argmax(prediction)
        result = intToNote[index]

        predictionOutput.append(result)

        pattern = numpy.append(pattern, index)

        pattern = pattern[1:len(pattern)]

    return predictionOutput


def createMidi(predictionOutput):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    outputNotes = []



    #prebacujemo vrednosti u midi fajl
    #prvo je potrebno napraviti objekat sa notama i akordima i njihovim 
    #vrednostima koje je generisao model
    #treba voditi racuna o offset-u da se note ne bi "lepile" tj da bi svaka
    #se svirale jedna nakon druge
    #to ne vazi za akord jer mozemo primetiti da je akord "skup" razlicitih nota 
    #koje se sviraju u istom trenutku tako da kada su tonovi iz akorda u pitanju
    #offset se ne menja da se taj akord ne bi "razbio"
    #takodje treba obratiti paznju i na odabir instrumenta(u nasem slucaju klavir)
    for pattern in predictionOutput:
        # akorde smo zapisivali kao tonove razdvojene tackama pa ih sada
        #pomocu . i prepoznajemo
        if ('.' in pattern) or pattern.isdigit():
            notesInChord = pattern.split('.')
            notes = []
            for currentNote in notesInChord:
                newNote = note.Note(int(currentNote))
                newNote.storedInstrument = instrument.Piano()
                notes.append(newNote)
            newChord = chord.Chord(notes)
            newChord.offset = offset
            outputNotes.append(newChord)
        # ovo je grana za "same" note
        else:
            newNote = note.Note(pattern)
            newNote.offset = offset
            newNote.storedInstrument = instrument.Piano()
            outputNotes.append(newNote)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    write(outputNotes)


#def generate():
    
#    print('Pokrenulo se')

#    model, networkInput, pitchNames, nDiff = trainNetwork()
#    predictionOutput = generateNotes(model, networkInput, pitchNames, nDiff)
#    createMidi(predictionOutput)

    



    
print('Pokrenulo se')

model, networkInput, pitchNames, nDiff = trainNetwork()



print(networkInput.shape)

predictionOutput = generateNotes(model, networkInput, pitchNames, nDiff)








prediction = model.predict(predictionInput, verbose = 1)

prediction


intToNote = dict((number, note) for number, note in enumerate(pitchNames))
pattern = []
for i in range(len(prediction[0])):
    pattern.append(intToNote[i])
    
print(pattern)

createMidi(pattern)


